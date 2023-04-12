import collections
import itertools

import numpy as np

import torch
from torch import nn

from efg.data.augmentations3d import _dict_select
from efg.geometry import box_ops_torch as box_ops
from efg.modeling.backbones.configurable_rpn import RPN
from efg.modeling.backbones.sparse_net import SpMiddleResNetFHD
from efg.modeling.readers.voxel_reader import VoxelMeanFeatureExtractor

from center_head import CenterHead
from center_utils import draw_umich_gaussian, gaussian_radius


class VoxelNet(nn.Module):
    def __init__(self, config, **kwargs):
        super(VoxelNet, self).__init__()

        self.config = config
        self.device = torch.device(config.model.device)

        self.reader = VoxelMeanFeatureExtractor(**config.model.reader)
        self.backbone = SpMiddleResNetFHD(**config.model.backbone)
        self.neck = RPN(config.model.neck)
        self.center_head = CenterHead(config)

        # Label Assignment
        assigner_config = config.model.loss
        self.out_size_factor = assigner_config.out_size_factor

        self.tasks = config.model.head.tasks
        self.gaussian_overlap = assigner_config.gaussian_overlap
        self._max_objs = assigner_config.max_objs
        self._min_radius = assigner_config.min_radius

        self.class_names_plain = list(itertools.chain(*[t["class_names"] for t in self.tasks]))

        self.to(self.device)

    def assign_one(self, datas, info, data_id):
        max_objs = self._max_objs
        class_names_by_task = [t["class_names"] for t in self.tasks]
        num_classes_by_task = [t["num_classes"] for t in self.tasks]

        # Calculate output featuremap size
        grid_size = datas["shape"][data_id]
        pc_range = datas["range"][data_id]
        voxel_size = datas["size"][data_id]

        feature_map_size = grid_size[:2] // self.out_size_factor
        example = {}

        gt_dict = info["annotations"]
        gt_boxes_mask = np.array([n in self.class_names_plain for n in gt_dict["gt_names"]], dtype=np.bool_)
        _dict_select(gt_dict, gt_boxes_mask)

        gt_classes = np.array([self.class_names_plain.index(n) + 1 for n in gt_dict["gt_names"]], dtype=np.int32)
        gt_dict["gt_classes"] = gt_classes

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in class_names_by_task:
            task_masks.append([np.where(gt_dict["gt_classes"] == class_name.index(i) + 1 + flag) for i in class_name])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        task_names = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            task_name = []
            for m in mask:
                task_box.append(gt_dict["gt_boxes"][m])
                task_class.append(gt_dict["gt_classes"][m] - flag2)
                task_name.append(gt_dict["gt_names"][m])
            task_boxes.append(np.concatenate(task_box, axis=0))
            task_classes.append(np.concatenate(task_class))
            task_names.append(np.concatenate(task_name))
            flag2 += len(mask)

        for task_box in task_boxes:
            # limit rad to [-pi, pi]
            task_box[:, -1] = box_ops.limit_period(task_box[:, -1], offset=0.5, period=np.pi * 2)

        # print(gt_dict.keys())
        gt_dict["gt_classes"] = task_classes
        gt_dict["gt_names"] = task_names
        gt_dict["gt_boxes"] = task_boxes

        draw_gaussian = draw_umich_gaussian
        hms, anno_boxs, inds, masks, cats = [], [], [], [], []
        for idx, task in enumerate(self.tasks):
            hm = np.zeros((len(class_names_by_task[idx]), feature_map_size[1], feature_map_size[0]), dtype=np.float32)

            # [reg, hei, dim, vx, vy, rots, rotc]
            anno_box = np.zeros((max_objs, 10), dtype=np.float32)
            ind = np.zeros((max_objs), dtype=np.int64)
            mask = np.zeros((max_objs), dtype=np.uint8)
            cat = np.zeros((max_objs), dtype=np.int64)
            num_objs = min(gt_dict["gt_boxes"][idx].shape[0], max_objs)
            for k in range(num_objs):
                cls_id = gt_dict["gt_classes"][idx][k] - 1
                L, W = gt_dict["gt_boxes"][idx][k][3], gt_dict["gt_boxes"][idx][k][4]
                L, W = L / voxel_size[0] / self.out_size_factor, W / voxel_size[1] / self.out_size_factor
                if L > 0 and W > 0:
                    radius = gaussian_radius((L, W), min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius))
                    # be really careful for the coordinate system of your box annotation.
                    x, y, z = (
                        gt_dict["gt_boxes"][idx][k][0],
                        gt_dict["gt_boxes"][idx][k][1],
                        gt_dict["gt_boxes"][idx][k][2],
                    )
                    coor_x, coor_y = (x - pc_range[0]) / voxel_size[0] / self.out_size_factor, (
                        y - pc_range[1]
                    ) / voxel_size[1] / self.out_size_factor
                    ct = np.array([coor_x, coor_y], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    # throw out not in range objects to avoid out of array area when creating the heatmap
                    if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(hm[cls_id], ct, radius)

                    new_idx = k
                    x, y = ct_int[0], ct_int[1]

                    cat[new_idx] = cls_id
                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1

                    vx, vy = gt_dict["gt_boxes"][idx][k][6:8]
                    rot = gt_dict["gt_boxes"][idx][k][-1]
                    anno_box[new_idx] = np.concatenate(
                        (
                            ct - (x, y),
                            z,
                            np.log(gt_dict["gt_boxes"][idx][k][3:6]),
                            np.array(vx),
                            np.array(vy),
                            np.sin(rot),
                            np.cos(rot),
                        ),
                        axis=None,
                    )

            hms.append(hm)
            anno_boxs.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            cats.append(cat)

        # used for two stage code
        boxes = flatten(gt_dict["gt_boxes"])
        classes = merge_multi_group_label(gt_dict["gt_classes"], num_classes_by_task)
        gt_boxes_and_cls = np.zeros((max_objs, 10), dtype=np.float32)
        boxes_and_cls = np.concatenate((boxes, classes.reshape(-1, 1).astype(np.float32)), axis=1)

        num_obj = len(boxes_and_cls)
        assert num_obj <= max_objs

        # x, y, z, w, l, h, rotation_y, velocity_x, velocity_y, class_name
        boxes_and_cls = boxes_and_cls[:, [0, 1, 2, 3, 4, 5, 8, 6, 7, 9]]
        gt_boxes_and_cls[:num_obj] = boxes_and_cls

        example.update(
            {
                "gt_boxes_and_cls": gt_boxes_and_cls,
                "hm": hms,
                "anno_box": anno_boxs,
                "ind": inds,
                "mask": masks,
                "cat": cats,
            }
        )

        return example

    def label_assign(self, datas, infos):
        targets_list = []
        for data_id, info in enumerate(infos):
            example = self.assign_one(datas, info, data_id)
            targets_list.append(example)
        return targets_list

    def forward(self, batched_inputs):
        """
        Data:   dict_keys(['voxels', 'coordinates', 'num_points_per_voxel', 'num_voxels', 'shape'])
        Infos:  dict_keys(['image', 'point_cloud', 'calib', 'annotations', 'root_path'])
        """

        with torch.no_grad():
            data = [bi[0] for bi in batched_inputs]
            infos = [bi[1] for bi in batched_inputs]

            datas = collate(data, self.device)

            voxels = datas["voxels"]
            coordinates = datas["coordinates"]
            num_points_in_voxel = datas["num_points_per_voxel"]
            num_voxels = datas["num_voxels"]
            input_shape = datas["shape"][0]
            batch_size = len(num_voxels)

        input_features = self.reader(voxels, num_points_in_voxel)
        x = self.backbone(input_features, coordinates, batch_size, input_shape)
        x = self.neck(x)

        preds = self.center_head(x)

        if self.training:
            with torch.no_grad():
                targets_list = self.label_assign(datas, infos)
                targets = collate(targets_list, self.device)
                targets.update(datas)
            return self.center_head.loss(targets, preds)
        else:
            return self.center_head.predict(datas, preds, self.config.model.post_process)


def collate(batch_list, device):
    targets_merged = collections.defaultdict(list)
    for targets in batch_list:
        for k, v in targets.items():
            targets_merged[k].append(v)

    batch_size = len(batch_list)
    ret = {}
    # centerpoint: 'hm', 'anno_box', 'ind', 'mask', 'cat'
    for key, elems in targets_merged.items():
        if key in ["voxels", "num_points_per_voxel", "num_voxels"]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0)).to(device)
        elif key in ["point_boxes", "image_boxes", "names", "difficulty", "group_ids", "gt_boxes_and_cls"]:
            max_gt = -1
            for k in range(batch_size):
                max_gt = max(max_gt, len(elems[k]))
                batch_gt_boxes3d = np.zeros((batch_size, max_gt, *elems[0].shape[1:]), dtype=elems[0].dtype)
            for i in range(batch_size):
                batch_gt_boxes3d[i, : len(elems[i])] = elems[i]
            if key != "names":
                batch_gt_boxes3d = torch.tensor(batch_gt_boxes3d)
            ret[key] = batch_gt_boxes3d
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ["coordinates", "points"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode="constant", constant_values=i)
                coors.append(coor_pad)
            ret[key] = torch.tensor(np.concatenate(coors, axis=0)).to(device)
        elif key in ["anchors", "reg_targets", "reg_weights", "labels", "anno_box", "hm", "cat", "mask", "ind"]:
            ret[key] = collections.defaultdict(list)
            res = []
            for elem in elems:
                for idx, ele in enumerate(elem):
                    ret[key][str(idx)].append(torch.tensor(ele))
            for kk, vv in ret[key].items():
                res.append(torch.stack(vv).to(device))
            ret[key] = res
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def merge_multi_group_label(gt_classes, num_classes_by_task):
    num_task = len(gt_classes)
    flag = 0
    for i in range(num_task):
        gt_classes[i] += flag
        flag += num_classes_by_task[i]
    return flatten(gt_classes)


def flatten(box):
    return np.concatenate(box, axis=0)
