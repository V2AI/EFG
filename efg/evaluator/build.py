import os

from efg.utils import distributed as comm

from .evaluator import DatasetEvaluators
from .registry import EVALUATORS


def build_evaluators(config, dataset=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    output_folder = os.path.join(config.trainer.output_dir, "inference")
    if not os.path.exists(output_folder):
        if comm.is_main_process():
            os.makedirs(output_folder)

    evaluator_list = []
    for evaluator_type in config.trainer.evaluators:
        evaluator = EVALUATORS.get(evaluator_type)(config, output_folder, dataset=dataset)
        evaluator_list.append(evaluator)

    return DatasetEvaluators(evaluator_list)
