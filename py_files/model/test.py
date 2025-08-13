import numpy as np
from joblib import load
import pandas as pd

# load your trained artifacts
svm = load("/Users/mattchew/30.007-model/py_files/model/svm_model_final.joblib")
le  = load("/Users/mattchew/30.007-model/py_files/model/label_encoder.joblib")



def predict_imfs(imfs_row):
    """imfs_row: iterable of length 9 [IMF_1..IMF_9]"""
    x = np.asarray(imfs_row, dtype=float).reshape(1, -1)

    # prediction
    idx = svm.predict(x)[0]               # integer class id
    label = le.inverse_transform([idx])[0]

    # SVM confidence-ish scores:
    # - binary: decision_function -> margin (sign decides class)
    # - multi: one-vs-one margins per pair
    margin = svm.decision_function(x)

    # if you calibrated with probability=True or CalibratedClassifierCV, you could use predict_proba
    # probs = svm.predict_proba(x)

    return int(idx), str(label), margin

# # example
# idx, label, margin = predict_imfs([0.144684889, 0.115587815, 0.127970684, 0.187782027, 0.179412217, 0.102710669, 0.040254408, 0.021612435, 0.013164505])
# print("idx:", idx, "label:", label, "margin:", margin)
# print("label map (idx→name):", dict(enumerate(le.classes_)))