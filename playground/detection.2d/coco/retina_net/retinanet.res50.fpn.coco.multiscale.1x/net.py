from retinanet import RetinaNet


def build_model(self, config):
    model = RetinaNet(config)

    return model
