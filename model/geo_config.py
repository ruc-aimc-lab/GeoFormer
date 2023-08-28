from yacs.config import CfgNode as CN

def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


_CN = CN()
_CN.LAYER_NAMES = ['self', 'cross'] * 2

_CN.NHEAD = 4
_CN.COARSE_THR = 0.2
_CN.FINE_TEMPERATURE = 0.1
_CN.FINE_THR = 0.1
_CN.WINDOW_SIZE = 5
_CN.TOPK = 1

default_cfg = lower_config(_CN)

# -----------------------------------------------------

def get_cfg_model():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()



