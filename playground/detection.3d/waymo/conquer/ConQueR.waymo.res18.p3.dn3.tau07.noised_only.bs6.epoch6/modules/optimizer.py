import collections.abc

import torch

from efg.solver import OPTIMIZERS

from .utils import filter_grads


def get_parameters(module, lr_multi=None, lr_module=[], lr_except=["backbone"]):
    param_optimizer = list(module.named_parameters())
    optimizer_grouped_parameters = [
        {
            "params": filter_grads(
                [p for n, p in param_optimizer if not any(nd in n for nd in (lr_except + lr_module))]
            ),
        },
        {
            "params": filter_grads(
                [
                    p
                    for n, p in param_optimizer
                    if any(nd in n for nd in lr_module) and not any(nd in n for nd in lr_except)
                ]
            ),
            "lr_multi": lr_multi if lr_multi is not None else 1.0,
        },
    ]

    return optimizer_grouped_parameters


@OPTIMIZERS.register()
class AdamWMulti:
    @staticmethod
    def build(cfg, model):
        backbone_groups = []
        transformer_groups = []
        backbone_param_group = {"params": filter_grads(model.backbone.parameters())}
        backbone_groups.append(backbone_param_group)
        transformer_param_group = get_parameters(
            model,
            lr_multi=cfg.solver.deform_lr_multi,
            lr_module=["linear_box"],
            lr_except=["backbone"],
        )
        transformer_groups.extend(transformer_param_group)
        model_params = (backbone_groups, transformer_groups)

        optim_config = cfg.solver.optimizer

        if isinstance(model_params[0], collections.abc.Sequence):
            param_groups = []
            backbone_group, other_group = model_params

            lr_backbone = optim_config.pop("lr_backbone", optim_config["lr"])

            for group in backbone_group:
                group["lr"] = lr_backbone
                param_groups.append(group)

            for group in other_group:
                if "lr_multi" in group:
                    group["lr"] = optim_config["lr"] * group.pop("lr_multi")
                param_groups.append(group)
        else:
            param_groups = [{"lr": optim_config["lr"], "params": model_params}]

        optimizer = torch.optim.AdamW(param_groups, **optim_config)
        return optimizer
