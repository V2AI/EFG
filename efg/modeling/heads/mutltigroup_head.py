from torch import nn


class Head(nn.Module):
    def __init__(
        self,
        num_input,
        num_pred,
        num_cls,
        use_dir=False,
        num_dir=0,
        header=True,
        norm=None,
        name="",
        focal_loss_init=False,
        **kwargs,
    ):
        super(Head, self).__init__(**kwargs)
        self.use_dir = use_dir

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)

        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)

    def forward(self, x):
        # ret_list = []
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_preds

        return ret_dict


class MultiGroupHead(nn.Module):
    def __init__(
        self,
        cfg,
        box_code_sizes,
        encode_background_as_zeros,
        reg_class_agnostic=False,
    ):
        super(MultiGroupHead, self).__init__()

        self.use_dir = cfg.MODEL.HEAD.LOSS_AUX.get("ENABLED", True)

        tasks = cfg.MODEL.HEAD.TASKS
        num_classes = [len(t["class_names"]) for t in tasks]
        self.num_anchor_per_locs = [2 * n for n in num_classes]
        self.norm_cfg = cfg.MODEL.HEAD.get("NORM", None)

        self.in_channels = cfg.MODEL.HEAD.IN_CHANNES

        num_clss = []
        num_preds = []
        num_dirs = []

        for num_c, num_a, box_cs in zip(num_classes, self.num_anchor_per_locs, box_code_sizes):
            if encode_background_as_zeros:
                num_cls = num_a * num_c
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)

            num_pred = num_a * box_cs
            num_preds.append(num_pred)

            num_dir = num_a * 2
            num_dirs.append(num_dir)

        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(
                Head(
                    self.in_channels,
                    num_pred,
                    num_cls,
                    use_dir=self.use_dir,
                    num_dir=num_dirs[task_id],
                    header=False,
                    norm=self.norm_cfg,
                )
            )

    def forward(self, x):
        ret_dicts = []
        for task in self.tasks:
            ret_dicts.append(task(x))

        return ret_dicts
