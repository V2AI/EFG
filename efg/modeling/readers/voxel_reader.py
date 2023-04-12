import torch
from torch import nn

from ..registry import READERS


@READERS.register()
class VoxelMeanFeatureExtractor(nn.Module):
    def __init__(self, num_input_features, norm="BN1d"):
        super(VoxelMeanFeatureExtractor, self).__init__()

        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, : self.num_input_features].sum(dim=1, keepdim=False) / num_voxels.type_as(
            features
        ).view(-1, 1)

        return points_mean.contiguous()


@READERS.register()
class DynamicMeanVFE(nn.Module):
    def __init__(self, num_input_features, voxel_size, point_cloud_range):
        super(DynamicMeanVFE, self).__init__()
        self.num_point_features = num_input_features

        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

        grid_size = (self.point_cloud_range[3:] - self.point_cloud_range[:3]) / self.voxel_size
        grid_size = grid_size.int()

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xyz = grid_size[0] * grid_size[1] * grid_size[2]
        self.scale_yz = grid_size[1] * grid_size[2]
        self.scale_z = grid_size[2]

        self.grid_size = grid_size

    def get_output_feature_dim(self):
        return self.num_point_features

    @torch.no_grad()
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                voxels: (num_voxels, max_points_per_voxel, C)
                voxel_num_points: optional (num_voxels)
        Returns:
            vfe_features: (num_voxels, C)
        """
        # batch_size = batch_dict['batch_size']
        points = batch_dict["points"]  # (batch_idx, x, y, z, i, e)

        # debug
        point_coords = torch.floor((points[:, 1:4] - self.point_cloud_range[0:3]) / self.voxel_size).int()
        mask = ((point_coords >= 0) & (point_coords < self.grid_size)).all(dim=1)
        points = points[mask]
        point_coords = point_coords[mask]
        merge_coords = (
            points[:, 0].int() * self.scale_xyz
            + point_coords[:, 0] * self.scale_yz
            + point_coords[:, 1] * self.scale_z
            + point_coords[:, 2]
        )
        points_data = points[:, 1:].contiguous()

        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True)

        from torch_scatter import scatter_mean

        points_mean = scatter_mean(points_data, unq_inv, dim=0)

        unq_coords = unq_coords.int()
        voxel_coords = torch.stack(
            (
                unq_coords // self.scale_xyz,
                (unq_coords % self.scale_xyz) // self.scale_yz,
                (unq_coords % self.scale_yz) // self.scale_z,
                unq_coords % self.scale_z,
            ),
            dim=1,
        )
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict["voxel_features"] = points_mean.contiguous()
        batch_dict["voxel_coords"] = voxel_coords.contiguous()

        return batch_dict
