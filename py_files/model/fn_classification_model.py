# dependencies
import numpy as np
from joblib import load
from pathlib import Path

# constants
from py_files.fn_cfg import SVM_MODEL_PATH, LABEL_ENCODER_PATH

class_map = {0: "Sitting", 1: "Standing"}
_svm = None
_le  = None

def _load():
    global _svm, _le
    if _svm is not None:  # already loaded
        return
    here = Path(__file__).resolve().parent        # .../py_files/model
    _svm = load(here / SVM_MODEL_PATH)
    _le  = load(here / LABEL_ENCODER_PATH)

def classify_imfs(imfs_row):
    """
    imfs_row: iterable of length 9 (IMF_1..IMF_9)
    returns: (idx:int, label:str, margin:np.ndarray or float)
    """
    if len(imfs_row) == 0:
        return None, None, None  # no input to predict
    else:
        _load()
        x = np.asarray(imfs_row, dtype=np.float32).reshape(1, -1)
        idx = int(_svm.predict(x)[0])                  # integer class id
        label = str(_le.inverse_transform([idx])[0])   # human label
        margin = _svm.decision_function(x)             # SVM margins
        try:
            margin = margin.tolist()                   # JSON-friendly
        except Exception:
            pass
        return idx, label, margin
