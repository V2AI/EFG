import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import zeros_, kaiming_normal_


class PointNet(nn.Module):
    def __init__(self, input_dim, joint_feat=False, channels=None):
        super(PointNet, self).__init__()
        self.joint_feat = joint_feat

        times = 1
        self.feat = PointNetfeat(input_dim, 1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, channels)

        self.pre_bn = nn.BatchNorm1d(input_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()

        self.fc_s1 = nn.Linear(channels * times, 256)
        self.fc_s2 = nn.Linear(256, 3, bias=False)
        self.fc_ce1 = nn.Linear(channels * times, 256)
        self.fc_ce2 = nn.Linear(256, 3, bias=False)
        self.fc_hr1 = nn.Linear(channels * times, 256)
        self.fc_hr2 = nn.Linear(256, 1, bias=False)
        self.init_weights()

    def forward(self, x, feat=None):
        if self.joint_feat:
            if len(feat.shape) > 2:
                feat = torch.max(feat, 2, keepdim=True)[0]
                x = feat.view(-1, self.output_channel)
                x = F.relu(self.bn1(self.fc1(x)))
                feat = F.relu(self.bn2(self.fc2(x)))
            else:
                feat = feat
            feat_traj = None
        else:
            x, feat_traj = self.feat(self.pre_bn(x))
            x = F.relu(self.bn1(self.fc1(x)))
            feat = F.relu(self.bn2(self.fc2(x)))

        return feat, feat_traj

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)


class MotionEncoder(nn.Module):
    def __init__(
        self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None
    ):
        super().__init__()
        self.pre_mlps = build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False,
        )
        self.mlps = build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False,
        )

        if out_channels is not None:
            self.out_mlps = build_mlps(
                c_in=hidden_dim,
                mlp_channels=[hidden_dim, hidden_dim, out_channels],
                ret_before_act=True,
            )
        else:
            self.out_mlps = None

    def forward(self, polylines, polylines_mask):
        """
        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):
        Returns:
        """
        batch_size, num_polylines, num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(
            batch_size,
            num_polylines,
            num_points_each_polylines,
            polylines_feature_valid.shape[-1],
        )
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat(
            (
                polylines_feature,
                pooled_feature[:, :, None, :].repeat(
                    1, 1, num_points_each_polylines, 1
                ),
            ),
            dim=-1,
        )

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(
            batch_size,
            num_polylines,
            num_points_each_polylines,
            polylines_feature_valid.shape[-1],
        )
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[
            0
        ]  # (batch_size, num_polylines, C)

        # out-mlp
        if self.out_mlps is not None:
            valid_mask = polylines_mask.sum(dim=-1) > 0
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(
                batch_size, num_polylines, feature_buffers_valid.shape[-1]
            )
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers


class PointNetfeat(nn.Module):
    def __init__(self, input_dim, x=1, outchannel=512):
        super(PointNetfeat, self).__init__()
        if outchannel == 256:
            self.output_channel = 256
        else:
            self.output_channel = 512 * x
        self.conv1 = torch.nn.Conv1d(input_dim, 64 * x, 1)
        self.conv2 = torch.nn.Conv1d(64 * x, 128 * x, 1)
        self.conv3 = torch.nn.Conv1d(128 * x, 256 * x, 1)
        self.conv4 = torch.nn.Conv1d(256 * x, self.output_channel, 1)
        self.bn1 = nn.BatchNorm1d(64 * x)
        self.bn2 = nn.BatchNorm1d(128 * x)
        self.bn3 = nn.BatchNorm1d(256 * x)
        self.bn4 = nn.BatchNorm1d(self.output_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x_ori = self.bn4(self.conv4(x))

        x = torch.max(x_ori, 2, keepdim=True)[0]

        x = x.view(-1, self.output_channel)
        return x, x_ori


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()])
            else:
                layers.extend(
                    [
                        nn.Linear(c_in, mlp_channels[k], bias=False),
                        nn.BatchNorm1d(mlp_channels[k]),
                        nn.ReLU(),
                    ]
                )
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)
