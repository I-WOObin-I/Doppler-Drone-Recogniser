import math
import random
from typing import Dict, Tuple, Any
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

HPARAM_SWEEP_LIN = 1
HPARAM_SWEEP_LOG = 2


def log_uniform(min_v: float, max_v: float, base: float = 10.0, rng: random.Random | None = None) -> float:
    """
    Sample a positive scalar log-uniformly between min_v and max_v.
    """
    if not (min_v > 0 and max_v > 0 and min_v < max_v):
        raise ValueError("For log-uniform sampling, require 0 < min_v < max_v.")
    r = rng or random
    lo = math.log(min_v, base)
    hi = math.log(max_v, base)
    return base ** r.uniform(lo, hi)


def lin_uniform(min_v: float, max_v: float, rng: random.Random | None = None) -> float:
    """
    Sample a positive scalar uniformly between min_v and max_v.
    """
    if not (min_v < max_v):
        raise ValueError("For uniform sampling, require min_v < max_v.")
    r = rng or random
    return r.uniform(min_v, max_v)


def sample_hparams(bounds: Dict[str, Tuple[list[float], int]], rng: random.Random | None = None) -> Dict[str, Any]:
    """
    Draw one random hparam configuration from provided bounds using utils.log_uniform / utils.lin_uniform.
    - For type == HPARAM_SWEEP_LOG: use log_uniform(min, max)
    - For type == HPARAM_SWEEP_LIN: use lin_uniform(min, max)
    Tries to pass rng to the utils functions if they accept it; falls back if not.
    """
    r = rng or random
    hparams: Dict[str, Any] = {}

    for name, (range_pair, sweep_type) in bounds.items():
        if not isinstance(range_pair, (list, tuple)) or len(range_pair) != 2:
            raise ValueError(f"Bounds for {name} must be a list/tuple [min, max]. Got: {range_pair}")
        lo, hi = float(range_pair[0]), float(range_pair[1])
        if sweep_type == HPARAM_SWEEP_LOG:
            value = log_uniform(lo, hi, rng=r)
        elif sweep_type == HPARAM_SWEEP_LIN:
            value = lin_uniform(lo, hi, rng=r)
        else:
            raise ValueError(f"Unknown sweep type for {name}: {sweep_type}")
        hparams[name] = value

    return hparams


def _log_confusion_matrix(self, labels, preds, label_names) -> plt.Figure:
        cm = confusion_matrix(labels, preds)
        fig = plt.figure(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names,
                    yticklabels=label_names)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        return fig