import collections
import math

import torch
from torch import nn

from losses import Det3DLoss
from modules.blocks import MLP
from modules.matcher import HungarianMatcher3d
from modules.metrics import Accuracy
from modules.utils import get_clones, inverse_sigmoid


class Det3DHead(nn.Module):
    def __init__(self, config, with_aux=False, with_metrics=False, num_classes=3, num_layers=1):
        super().__init__()
        # build detection heads for transformer layers
        hidden_dim = config.model.hidden_dim

        class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)
        bbox_embed = MLP(hidden_dim, hidden_dim, 7, 3)

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        class_embed.layers[-1].bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

        self.class_embed = get_clones(class_embed, num_layers)
        self.bbox_embed = get_clones(bbox_embed, num_layers)

        # build losses
        matcher_config = config.model.loss.matcher
        matcher = HungarianMatcher3d(
            cost_class=matcher_config.class_weight,
            cost_bbox=matcher_config.bbox_weight,
            cost_giou=matcher_config.giou_weight,
            cost_rad=matcher_config.rad_weight,
        )
        weight_dict = {
            "loss_ce": config.model.loss.class_loss_coef,
            "loss_bbox": config.model.loss.bbox_loss_coef,
            "loss_giou": config.model.loss.giou_loss_coef,
            "loss_rad": config.model.loss.rad_loss_coef,
        }
        losses = ["focal_labels", "boxes"]
        self.losses = Det3DLoss(
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
        )

        # setup aux loss weight
        if with_aux:
            aux_weight_dict = {}
            num_layers = config.model.transformer.dec_layers
            if hasattr(self.losses, "weight_dict"):
                aux_weight_dict.update({k + "_enc_0": v for k, v in self.losses.weight_dict.items()})
                for i in range(num_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in self.losses.weight_dict.items()})
                self.losses.weight_dict.update(aux_weight_dict)

        if with_metrics:
            # build metrics
            if not isinstance(config.model.metrics, collections.abc.Sequence):
                metrics = (config.model.metrics,)
            else:
                metrics = config.model.metrics
            module_metrics = {}
            for metric in metrics:
                module_metric = Accuracy(**metric["params"])
                module_metrics[metric["type"]] = module_metric
            self.metrics = module_metrics
        self.config = config

    def forward(self, embed, anchors, layer_idx=0):
        cls_logits = self.class_embed[layer_idx](embed)
        box_coords = (self.bbox_embed[layer_idx](embed) + inverse_sigmoid(anchors)).sigmoid()
        return cls_logits, box_coords

    def compute_losses(self, outputs, targets, dn_meta=None):
        loss_dict = self.losses(outputs, targets, dn_meta=dn_meta)

        weight_dict = self.losses.weight_dict
        for k, v in loss_dict.items():
            if k in weight_dict:
                loss_dict[k] = v * weight_dict[k]

        if hasattr(self, "metrics"):
            for name, metric in self.metrics.items():
                if name == "accuracy":
                    loss_dict.update(metric(*self.losses.get_target_classes()))
                else:
                    loss_dict.update(metric(outputs, targets))

        return loss_dict
