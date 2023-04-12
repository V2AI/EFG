import collections
import json
import os
from ast import literal_eval

from omegaconf import OmegaConf

import torch


def load_yaml(file_path):
    mapping = OmegaConf.load(file_path)

    includes = mapping.get("includes", [])
    include_mapping = OmegaConf.create()

    for include in includes:
        include = os.path.join("./", include)
        current_include_mapping = load_yaml(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    mapping.pop("includes", None)
    mapping = OmegaConf.merge(include_mapping, mapping)

    return mapping


class Configuration:
    def __init__(self, args):
        self.config = {}
        self.args = args
        self._register_resolvers()

        default_config = self._build_default_config()
        user_config = self._build_user_config(args.config)

        self._default_config = default_config
        self._user_config = user_config
        self.config = OmegaConf.merge(default_config, user_config)

        self.config = self._merge_with_dotlist(self.config, args.opts)

    def _build_default_config(self):
        self.default_config_path = self._get_default_config_path()
        default_config = load_yaml(self.default_config_path)
        return default_config

    def _build_user_config(self, config_path):
        user_config = {}

        # Update user_config with opts if passed
        self.config_path = config_path
        if self.config_path is not None:
            user_config = load_yaml(self.config_path)

        return user_config

    def get_config(self):
        self._register_resolvers()
        return self.config

    def _register_resolvers(self):
        OmegaConf.clear_resolvers()
        # Device count resolver
        device_count = max(1, torch.cuda.device_count())
        OmegaConf.register_new_resolver("device_count", lambda: device_count)

    def _merge_with_dotlist(self, config, opts):
        # TODO: To remove technical debt, a possible solution is to use
        # struct mode to update with dotlist OmegaConf node. Look into this
        # in next iteration
        if opts is None:
            opts = []

        if len(opts) == 0:
            return config

        # Support equal e.g. model=visual_bert for better future hydra support
        has_equal = opts[0].find("=") != -1

        if has_equal:
            opt_values = [opt.split("=") for opt in opts]
        else:
            assert len(opts) % 2 == 0, "Number of opts should be multiple of 2"
            opt_values = zip(opts[0::2], opts[1::2])

        for opt, value in opt_values:
            splits = opt.split(".")
            current = config
            for idx, field in enumerate(splits):
                array_index = -1
                if field.find("[") != -1 and field.find("]") != -1:
                    stripped_field = field[: field.find("[")]
                    array_index = int(field[field.find("[") + 1 : field.find("]")])
                else:
                    stripped_field = field
                if stripped_field not in current:
                    raise AttributeError(
                        "While updating configuration option {} is missing from configuration at field {}".format(
                            opt, stripped_field
                        )
                    )
                if isinstance(current[stripped_field], collections.abc.Mapping):
                    current = current[stripped_field]
                elif isinstance(current[stripped_field], collections.abc.Sequence) and array_index != -1:
                    current_value = current[stripped_field][array_index]

                    # Case where array element to be updated is last element
                    if not isinstance(current_value, (collections.abc.Mapping, collections.abc.Sequence)):
                        print("Overriding option {} to {}".format(opt, value))
                        current[stripped_field][array_index] = self._decode_value(value)
                    else:
                        # Otherwise move on down the chain
                        current = current_value
                else:
                    if idx == len(splits) - 1:
                        print("Overriding option {} to {}".format(opt, value))
                        current[stripped_field] = self._decode_value(value)
                    else:
                        raise AttributeError(
                            "While updating configuration option {} is not present after field {}".format(
                                opt, stripped_field
                            )
                        )

        return config

    def _decode_value(self, value):
        # https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L400
        if not isinstance(value, str):
            return value

        if value == "None":
            value = None

        try:
            value = literal_eval(value)
        except ValueError:
            pass
        except SyntaxError:
            pass
        return value

    def freeze(self):
        OmegaConf.set_struct(self.config, True)

    def defrost(self):
        OmegaConf.set_struct(self.config, False)

    def _convert_node_to_json(self, node):
        container = OmegaConf.to_container(node, resolve=True)
        return json.dumps(container, indent=4, sort_keys=True)

    def _get_default_config_path(self):
        directory = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(directory, "..", "config", "default.yaml")
