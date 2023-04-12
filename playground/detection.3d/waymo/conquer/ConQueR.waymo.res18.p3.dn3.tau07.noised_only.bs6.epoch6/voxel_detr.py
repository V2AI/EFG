import copy

import torch
from torch import nn

from efg.data.datasets.waymo import collate
from efg.modeling.backbones.fpn import build_resnet_fpn_backbone
from efg.modeling.readers.voxel_reader import VoxelMeanFeatureExtractor

from cdn import dn_post_process, prepare_for_cdn
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
            num_classes=self.num_classes,
            mom=config.model.contrastive.mom,
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

        self.transformer.decoder_gt = copy.deepcopy(self.transformer.decoder)
        for param_q, param_k in zip(self.transformer.decoder.parameters(), self.transformer.decoder_gt.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # build annotaion coder
        self.box_coder = VoxelBoxCoder3D(
            config.dataset.voxel_size,
            config.dataset.pc_range,
            device=self.device,
        )

        # contrastive projector
        contras_dim = config.model.contrastive.dim
        self.eqco = config.model.contrastive.eqco
        self.tau = config.model.contrastive.tau
        self.contras_loss_coeff = config.model.contrastive.loss_coeff
        self.projector = nn.Sequential(
            nn.Linear(10, contras_dim),
            nn.ReLU(),
            nn.Linear(contras_dim, contras_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(contras_dim, contras_dim),
            nn.ReLU(),
            nn.Linear(contras_dim, contras_dim),
        )
        self.similarity_f = nn.CosineSimilarity(dim=2)

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

        dn = self.config.model.dn
        if self.training and dn.enabled and dn.dn_number > 0:
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(targets, dn.dn_number, dn.dn_label_noise_ratio, dn.dn_box_noise_scale),
                training=self.training,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                label_enc=None,
            )
        else:
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        outputs = self.transformer(
            features,
            pos_encodings,
            input_query_bbox,
            input_query_label,
            attn_mask,
            targets=targets,
        )
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

        # dn post process
        if dn.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord = dn_post_process(
                outputs_class,
                outputs_coord,
                dn_meta,
                self.aux_loss,
                self._set_aux_loss,
            )

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
                "pred_logits": outputs_class[-1][:, : self.num_queries],
                "pred_boxes": outputs_coord[-1][:, : self.num_queries],
                "aux_outputs": self._set_aux_loss(
                    outputs_class[:-1, :, : self.num_queries], outputs_coord[:-1, :, : self.num_queries]
                ),
            }
            dec_losses = self.transformer.decoder.detection_head.compute_losses(outputs, targets, dn_meta)
            losses.update(dec_losses)

            # compute contrastive loss
            per_gt_num = [tgt["gt_boxes"].shape[0] for tgt in targets]
            max_gt = max(per_gt_num)
            num_gts = sum(per_gt_num)
            if num_gts > 0:
                for li in range(self.config.model.transformer.dec_layers):
                    contrastive_loss = 0.0
                    projs = torch.cat((outputs_class[li], outputs_coord[li]), dim=-1)
                    gt_projs = self.projector(projs[:, self.num_queries :].detach())
                    pred_projs = self.predictor(self.projector(projs[:, : self.num_queries]))
                    # num_gts x num_locs
                    pos_idxs = list(range(1, dn_meta["num_dn_group"] + 1))
                    for bi, idx in enumerate(outputs["matched_indices"]):
                        sim_matrix = (
                            self.similarity_f(
                                gt_projs[bi].unsqueeze(1),
                                pred_projs[bi].unsqueeze(0),
                            )
                            / self.tau
                        )
                        matched_pairs = torch.stack(idx, dim=-1)
                        neg_mask = projs.new_ones(self.num_queries).bool()
                        neg_mask[matched_pairs[:, 0]] = False
                        for pair in matched_pairs:
                            pos_mask = torch.tensor([int(pair[1] + max_gt * pi) for pi in pos_idxs], device=self.device)
                            pos_pair = sim_matrix[pos_mask, pair[0]].view(-1, 1)
                            neg_pairs = sim_matrix[:, neg_mask][pos_mask]
                            loss_gti = (
                                torch.log(torch.exp(pos_pair) + torch.exp(neg_pairs).sum(dim=-1, keepdim=True))
                                - pos_pair
                            )
                            contrastive_loss += loss_gti.mean()
                    losses[f"loss_contrastive_dec_{li}"] = self.contras_loss_coeff * contrastive_loss / num_gts

            return losses
        else:
            out_logits = outputs_class[-1][:, : self.num_queries]
            out_bbox = outputs_coord[-1][:, : self.num_queries]

            out_prob = out_logits.sigmoid()
            out_prob = out_prob.view(out_logits.shape[0], -1)
            out_bbox = self.box_coder.decode(out_bbox)

            def _process_output(indices, bboxes):
                topk_boxes = indices.div(out_logits.shape[2], rounding_mode="floor")
                labels = indices % out_logits.shape[2]
                boxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, out_bbox.shape[-1]))
                return labels + 1, boxes, topk_boxes

            topk_indices = torch.nonzero(out_prob >= 0.1, as_tuple=True)[1]
            scores = out_prob[:, topk_indices]

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
