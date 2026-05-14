"""
predict.py — Нақты уақыт болжауы (RF balanced_subsample + tuned)
"""
import pickle, warnings
from pathlib import Path
import numpy as np
from dataset import FEATURES

warnings.filterwarnings("ignore")
MODELS_DIR   = Path("models")
BEST_MODEL   = "rf_bs_tuned"
_LOADED: dict = {}

def load_best_model():
    if not _LOADED:
        path = MODELS_DIR / f"{BEST_MODEL}.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}\nRun: python train.py")
        with open(path,"rb") as f: _LOADED.update(pickle.load(f))
    return _LOADED

def predict_single(sample: dict) -> dict:
    art   = load_best_model()
    model = art["model"]
    sc    = art.get("scaler")
    thresh= art.get("threshold", 0.5)
    x = np.array([sample.get(f,0.0) for f in FEATURES], dtype=float).reshape(1,-1)
    if sc is not None: x = sc.transform(x)
    prob = model.predict_proba(x)[0][1]
    pred = int(prob >= thresh)
    fi   = art.get("feature_importance",{})
    feats= sorted(fi.items(), key=lambda t:t[1], reverse=True)[:5]
    return {
        "prediction": pred,
        "label": "attack" if pred==1 else "normal",
        "probability": round(float(prob),4),
        "risk_score": round(float(prob)*100,2),
        "alert": pred==1,
        "threshold_used": thresh,
        "top_risk_features": [{"feature":f,"value":round(v,4)} for f,v in feats],
    }

def get_model_info() -> dict:
    art = load_best_model()
    return {
        "model_name": BEST_MODEL,
        "model_type": "RandomForestClassifier (balanced_subsample + tuned threshold)",
        "threshold": art.get("threshold",0.5),
        "metrics": art.get("metrics",{}),
        "features": FEATURES,
        "n_features": len(FEATURES),
        "feature_importance": art.get("feature_importance",{}),
    }
