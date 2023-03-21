from voxelnet import VoxelNet


def build_model(self, config):
    model = VoxelNet(config)
    return model
