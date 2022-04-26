import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import CityscapesInstanceEvaluator, verify_results

from swint.config import add_swint_config

from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params

class Trainer(DefaultTrainer):
  @classmethod
  def build_evaluator(cls, cfg, dataset_name):
      return super().build_evaluator(cfg, dataset_name)



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_swint_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg