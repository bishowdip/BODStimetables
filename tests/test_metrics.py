import numpy as np
import pandas as pd

from src.ml.evaluate import _curve_points


def test_curve_points_perfect_separation():
    # scores rank all positives above all negatives -> ROC reaches (0,1) corner
    pdf = pd.DataFrame({"label": [0, 0, 1, 1], "score": [0.1, 0.2, 0.8, 0.9]})
    roc, pr = _curve_points(pdf)
    assert roc.tpr.max() == 1.0
    assert roc.fpr.max() == 1.0
    # at the point where all positives are recovered, fpr should still be 0
    assert (roc[roc.tpr == 1.0].fpr.min()) == 0.0
    assert pr.precision.iloc[0] == 1.0  # top-ranked item is a true positive


def test_curve_points_endpoints():
    pdf = pd.DataFrame({"label": [0, 1, 0, 1], "score": [0.5, 0.4, 0.3, 0.2]})
    roc, _ = _curve_points(pdf)
    assert roc.iloc[0].tolist() == [0.0, 0.0]   # curve starts at origin
    assert np.isclose(roc.iloc[-1].fpr, 1.0)
    assert np.isclose(roc.iloc[-1].tpr, 1.0)
