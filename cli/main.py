import argparse
import os
import sys
from os.path import dirname

from omegaconf import DictConfig, ListConfig

import torch

import efg
from efg.config import Configuration
from efg.data import build_dataloader, build_dataset, seed_all_rng
from efg.engine import build_trainer
from efg.engine.launch import launch, slurm_launch
from efg.evaluator import build_evaluators
from efg.utils import distributed as comm
from efg.utils.collect_env import collect_env_info
from efg.utils.file_io import PathManager
from efg.utils.logger import setup_logger


def format_dict_config(dc, indent=0):
    GREEN = '\033[92m'  # ANSI escape sequence for green
    YELLOW = '\033[93m'  # ANSI escape sequence for yellow
    END = '\033[0m'  # ANSI escape sequence to reset color

    formatted_str = ""
    if isinstance(dc, DictConfig):
        for key, value in dc.items():
            indent_str = '' if indent == 0 else (" " * (4 * (indent - 1)) + "|-- ")
            formatted_str += indent_str + GREEN + str(key) + END + ": "
            if isinstance(value, ListConfig):
                is_complex_list = any(isinstance(item, DictConfig) for item in value)
                if is_complex_list:
                    formatted_str += "\n"
                    for item in value:
                        if isinstance(item, DictConfig):
                            formatted_str += format_dict_config(item, indent + 1)
                        else:
                            list_indent_str = " " * (4 * indent) + "|-- "
                            formatted_str += list_indent_str + YELLOW + str(item) + END + "\n"
                else:
                    formatted_str += YELLOW + "[" + ", ".join(map(str, value)) + "]" + END + "\n"
            elif isinstance(value, list):
                formatted_str += YELLOW + "[" + ", ".join(map(str, value)) + "]" + END + "\n"
            elif isinstance(value, DictConfig):
                formatted_str += "\n" + format_dict_config(value, indent + 1)
            else:
                formatted_str += YELLOW + str(value) + END + "\n"
    else:
        formatted_str += YELLOW + str(dc) + END + "\n"
    return formatted_str


def get_parser():
    parser = argparse.ArgumentParser("EFG default argument parser")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file.")

    parser.add_argument("--launcher", type=str, default="pytorch")  # option slurm
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")
    parser.add_argument("--master-port", type=int, default=12345)
    parser.add_argument("--dist-url", default="auto")

    parser.add_argument("--resume", action="store_true", help="whether to attempt to resume from the checkpoint")
    parser.add_argument("--debug", action="store_true", help="debug mode")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options from command line")

    return parser


def link_log(output_dir, link_name="log"):
    """
    Thus function assumes that the user are currently at the experiments' directories.

    Create a softlink to output dir.
    Args:
        link_name(str): name of softlink
    """
    if os.path.islink(link_name) and os.readlink(link_name) != output_dir:
        os.system("rm " + link_name)
    if not os.path.exists(link_name):
        cmd = "ln -s {} {}".format(output_dir, link_name)
        os.system(cmd)


def worker(args):
    configuration = Configuration(args)
    config = configuration.get_config()

    # setup global logger
    output_dir = os.path.join(config.trainer.output_dir, "EFG", os.getcwd().split("playground")[1][1:])
    if comm.is_main_process() and output_dir:
        PathManager.mkdirs(output_dir)
        link_log(output_dir)
    config.trainer.output_dir = output_dir

    logger = setup_logger(output_dir, distributed_rank=comm.get_rank())

    logger.info(f"Command Line Args:\n{args}")
    logger.info(f"Environment info:\n{collect_env_info()}")

    # if we manually set the random seed
    if config.misc.seed >= 0:
        manual_set_generator = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        manual_set_generator = False

    torch.backends.cudnn.benchmark = config.misc.cudnn_benchmark

    seed = seed_all_rng(None if config.misc.seed < 0 else config.misc.seed)
    config.misc.seed = seed

    logger.info(f"Running with full config:\n\n{format_dict_config(config)}")

    from net import build_model  # net.py in experiment directories
    trainer = build_trainer(config, build_model)

    if config.task == "train":
        if args.resume:
            trainer.resume_or_load(args.resume)
        trainer.train()
        # Perform evaluation at the end of training
        config.task = "val"
        eval_dataset = build_dataset(config)
        eval_dataloader = build_dataloader(config, eval_dataset, msg=manual_set_generator)
        evaluators = build_evaluators(config, eval_dataset)
        trainer.evaluate(evaluators, eval_dataloader, test=False)
    elif config.task == "val" or config.task == "test":
        trainer.resume_or_load()
        evaluators = build_evaluators(config, trainer.dataset)
        trainer.evaluate(evaluators, test=config.task == "test")
    else:
        raise NotImplementedError


def main():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["EFG_PATH"] = dirname(dirname(efg.__file__))
    sys.path.insert(0, "./")

    parser = get_parser()
    args = parser.parse_args()

    if args.launcher == "pytorch":
        launcher = launch
    elif args.launcher == "slurm":
        launcher = slurm_launch

    launcher(
        worker,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        port=args.master_port,
        args=(args,),
    )


if __name__ == "__main__":
    main()
