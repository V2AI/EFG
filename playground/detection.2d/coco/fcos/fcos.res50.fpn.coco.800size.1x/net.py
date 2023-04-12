from fcos import FCOS


def build_model(self, config):
    model = FCOS(config)

    return model
