import torch.nn as nn

from .position_encoding import build_position_encoding


class Backbone3d(nn.Module):
    def __init__(
        self,
        hidden_dim,
        reader,
        extractor,
        position_encoding,
        out_features=[],
    ):
        super(Backbone3d, self).__init__()
        self.reader = reader
        self.extractor = extractor
        self.position_encoding = build_position_encoding(position_encoding, hidden_dim)
        self.out_features = out_features
        self.num_channels = [
            extractor.out_channels,
        ] * len(out_features)

    def forward(self, voxels, coordinates, num_points_per_voxel, batch_size, input_shape):
        encoded_input = self.reader(voxels, num_points_per_voxel, coordinates)
        backbone_features = self.extractor(encoded_input, coordinates, batch_size, input_shape)

        outputs = []
        for of in self.out_features:
            out = backbone_features[of]
            pos = self.position_encoding(out).type_as(out)
            outputs.append((out, pos))

        return outputs
