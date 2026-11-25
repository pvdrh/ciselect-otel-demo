"""D2-5: B4-LGBM và B4-LR (C6). Trả điểm cho từng (mutant, test) của fold dự đoán."""
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import features

N_JOBS = 2  # ponytail: giữ ít luồng để không làm nhiễu số đo thời gian của batch D1-7 đang chạy


def _model(kind, params, seed):
    if kind in ("lgbm", "lgbm_nofr"):
        return LGBMClassifier(**params, random_state=seed, deterministic=True, force_row_wise=True,
                              n_jobs=N_JOBS, verbose=-1)
    if kind == "rf":  # #51 thăm dò
        return RandomForestClassifier(**params, random_state=seed, n_jobs=N_JOBS)
    return make_pipeline(StandardScaler(), LogisticRegression(C=params["C"], class_weight=params.get("class_weight"),
                                                              max_iter=2000, random_state=seed))


def fit_predict(kind, params, d, train_ids, pred_ids, seed):
    drop = ("fail_rate",) if kind == "lgbm_nofr" else ()  # #47 Q1: phân tích phụ không có lịch sử fail
    X, ktr = features(d, train_ids, train_ids, drop)
    lab = {(r["mutant_id"], r["test_id"]): r["label"] for r in d.labels}
    y = np.array([lab[k] for k in ktr])
    Xp, kp = features(d, train_ids, pred_ids, drop)
    if y.min() == y.max():  # train chỉ có một lớp: điểm hằng
        return {k: float(y[0]) for k in kp}
    p = _model(kind, params, seed).fit(X, y).predict_proba(Xp)[:, 1]
    return {k: float(v) for k, v in zip(kp, p)}
