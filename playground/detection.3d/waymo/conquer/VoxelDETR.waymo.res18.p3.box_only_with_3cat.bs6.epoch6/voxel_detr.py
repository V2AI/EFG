import copy

import torch
from torch import nn

from efg.data.datasets.waymo import collate
from efg.modeling.backbones.fpn import build_resnet_fpn_backbone
from efg.modeling.readers.voxel_reader import VoxelMeanFeatureExtractor

from heads import Det3DHead
from modules.backbone3d import Backbone3d
from modules.box_coder import VoxelBoxCoder3D
from transformer import Transformer


class VoxelDETR(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.device = torch.device(config.model.device)

        # setup properties
        self.hidden_dim = config.model.hidden_dim
        self.aux_loss = config.model.aux_loss
        self.num_classes = len(config.dataset.classes)
        self.num_queries = config.model.transformer.num_queries

        # build backbone
        input_dim = len(config.dataset.format) if config.dataset.nsweeps == 1 else len(config.dataset.format) + 1
        reader = VoxelMeanFeatureExtractor(**config.model.backbone.reader, num_input_features=input_dim)
        extractor = build_resnet_fpn_backbone(config.model.backbone.extractor, input_dim)
        self.backbone = Backbone3d(
            config.model.backbone.hidden_dim,
            reader,
            extractor,
            config.model.backbone.position_encoding,
            out_features=config.model.backbone.out_features,
        )
        in_channels = self.backbone.num_channels

        # build input projection from backbone to transformer
        self.input_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels[i], self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
                for i in range(len(self.backbone.out_features))
            ]
        )
        for module in self.input_proj.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # build transformer
        self.transformer = Transformer(
            d_model=config.model.transformer.hidden_dim,
            nhead=config.model.transformer.nhead,
            nlevel=len(config.model.backbone.out_features),
            num_encoder_layers=config.model.transformer.enc_layers,
            num_decoder_layers=config.model.transformer.dec_layers,
            dim_feedforward=config.model.transformer.dim_feedforward,
            dropout=config.model.transformer.dropout,
            num_queries=config.model.transformer.num_queries,
        )
        self.transformer.proposal_head = Det3DHead(
            config,
            with_aux=False,
            with_metrics=False,
            num_classes=1,
            num_layers=1,
        )
        self.transformer.decoder.detection_head = Det3DHead(
            config,
            with_aux=True,
            with_metrics=True,
            num_classes=len(config.dataset.classes),
            num_layers=config.model.transformer.dec_layers,
        )

        # build annotaion coder
        self.box_coder = VoxelBoxCoder3D(
            config.dataset.voxel_size,
            config.dataset.pc_range,
            device=self.device,
        )

        self.config = config
        self.to(self.device)

    def forward(self, batched_inputs):
        batch_size = len(batched_inputs)

        # samples: ['voxels', 'points', 'coordinates', 'num_points_per_voxel', 'num_voxels', 'shape', 'range', 'size']
        samples = collate([bi[0] for bi in batched_inputs], self.device)

        if self.training:
            targets = [bi[1]["annotations"] for bi in batched_inputs]
            for key in ["gt_boxes", "difficulty", "num_points_in_gt", "labels"]:
                for i in range(batch_size):
                    targets[i][key] = torch.tensor(targets[i][key], device=self.device)
            targets = [self.box_coder.encode(tgt) for tgt in targets]
        else:
            targets = None

        voxels, coords, num_points_per_voxel, input_shape = (
            samples["voxels"],
            samples["coordinates"],
            samples["num_points_per_voxel"],
            samples["shape"][0],
        )

        ms_backbone_features_with_pos_embed = self.backbone(
            voxels, coords, num_points_per_voxel, batch_size, input_shape
        )

        features = []
        pos_encodings = []
        for idx, feat_pos in enumerate(ms_backbone_features_with_pos_embed):
            features.append(self.input_proj[idx](feat_pos[0]))
            pos_encodings.append(feat_pos[1])

        outputs = self.transformer(features, pos_encodings)
        hidden_state, init_reference, inter_references, src_embed, src_ref_windows, src_indexes = outputs

        # decoder
        outputs_classes = []
        outputs_coords = []
        for idx in range(hidden_state.shape[0]):
            if idx == 0:
                reference = init_reference
            else:
                reference = inter_references[idx - 1]
            outputs_class, outputs_coord = self.transformer.decoder.detection_head(hidden_state[idx], reference, idx)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if self.training:
            losses = {}
            # compute encoder losses
            enc_class, enc_coords = self.transformer.proposal_head(src_embed, src_ref_windows)
            bin_targets = copy.deepcopy(targets)
            [tgt["labels"].fill_(0) for tgt in bin_targets]
            enc_outputs = {
                "topk_indexes": src_indexes,
                "pred_logits": enc_class,
                "pred_boxes": enc_coords,
            }
            enc_losses = self.transformer.proposal_head.compute_losses(enc_outputs, bin_targets)
            losses.update({k + "_enc": v for k, v in enc_losses.items()})

            # compute decoder losses
            outputs = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
                "aux_outputs": self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1]),
            }
            dec_losses = self.transformer.decoder.detection_head.compute_losses(outputs, targets)
            losses.update(dec_losses)

            return losses
        else:
            # out_logits = outputs_class.squeeze()
            # out_bbox = outputs_coord.squeeze()

            out_logits = outputs_class[-1]
            out_bbox = outputs_coord[-1]

            out_prob = out_logits.sigmoid()
            out_prob = out_prob.view(out_logits.shape[0], -1)
            out_bbox = self.box_coder.decode(out_bbox)

            def _process_output(indices, bboxes):
                topk_boxes = indices.div(out_logits.shape[2], rounding_mode="floor")
                labels = indices % out_logits.shape[2]
                boxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_bbox.shape[-1]))
                return labels + 1, boxes, topk_boxes

            scores, topk_indices = torch.topk(out_prob, 300, dim=1, sorted=False)
            # # VEHICLE: 0.15, PEDESTRIAN: 0.12, CYCLIST: 0.1
            # topk_indices = torch.nonzero(out_prob >= 0.1, as_tuple=True)[1]
            # scores = out_prob[:, topk_indices]

            labels, boxes, topk_indices = _process_output(topk_indices.view(1, -1), out_bbox)

            results = [
                {
                    "scores": s.detach().cpu(),
                    "labels": l.detach().cpu(),
                    "boxes3d": b.detach().cpu(),
                }
                for s, l, b in zip(scores, labels, boxes)
            ]
            return results

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class, outputs_coord)]
