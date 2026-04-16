import time
from typing import Callable, Optional

import numpy as np

from py_files.model.fn_classification_model import classify_imfs, class_map


def classify_result(res: dict) -> dict:
    """
    Convert a detector result dict into a classifier output dict.
    Expected input:
      {"EnvProfile", "Duration_ms", "IMF", "t_abs_start"}
    """
    imfs = np.asarray(res["IMF"], dtype=np.float32)
    idx, label, margin = classify_imfs(imfs)

    try:
        t_wall = time.time()
    except Exception:
        t_wall = 0.0

    out = {
        "idx": int(idx) if idx is not None else -1,
        "label": str(label),
        "dur_ms": float(res.get("Duration_ms", 0.0)),
        "env": res.get("EnvProfile", "unknown"),
        "t": t_wall,
        "margin": margin,
        "IMF": res["IMF"],
    }
    return out


def print_classification(out: dict):
    print(
        f"State: {out['label']}, "
        f"Duration: {out['dur_ms']:.1f} ms, "
        f"Env: {out['env']}",
        flush=True,
    )


def make_segment_callback(
    handler: Optional[Callable[[dict], None]] = None,
) -> Callable[[dict], None]:
    """
    Returns a callback suitable for start_audio_pipeline(on_segment=...)
    """
    if handler is None:
        handler = print_classification

    def _cb(res: dict):
        out = classify_result(res)
        handler(out)

    return _cb