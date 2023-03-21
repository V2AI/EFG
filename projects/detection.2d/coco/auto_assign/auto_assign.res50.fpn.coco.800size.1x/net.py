from auto_assign import AutoAssign


def build_model(self, cfg):
    """
    Build AutoAssign

    Returns:
        an instance of :class:`AutoAssign`
    """

    model = AutoAssign(cfg)

    return model
