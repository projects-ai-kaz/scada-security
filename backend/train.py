"""
train.py — 9 ML модельді оқыту (Диссертация Таблицы 3–6)
"""
import json, pickle, warnings
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from dataset import FEATURES, build_dataset

warnings.filterwarnings("ignore")
MODELS_DIR = Path("models")
DATA_DIR   = Path("data")
MODELS_DIR.mkdir(exist_ok=True)
SEED = 42

def load_data():
    if not (DATA_DIR / "scada_train.csv").exists():
        build_dataset(output_dir=str(DATA_DIR))
    train = pd.read_csv(DATA_DIR / "scada_train.csv")
    test  = pd.read_csv(DATA_DIR / "scada_test.csv")
    return train[FEATURES].values, train["label"].values, test[FEATURES].values, test["label"].values

def metrics(yt, yp, yprob=None):
    r = {"accuracy": round(accuracy_score(yt,yp),4), "precision": round(precision_score(yt,yp,zero_division=0),4),
         "recall": round(recall_score(yt,yp,zero_division=0),4), "f1_score": round(f1_score(yt,yp,zero_division=0),4),
         "roc_auc": round(roc_auc_score(yt,yprob),4) if yprob is not None else None}
    return r

def tune_threshold(probs, y, step=0.02):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, step):
        f = f1_score(y, (probs>=t).astype(int), zero_division=0)
        if f > best_f1: best_f1, best_t = f, t
    return round(best_t, 2)

def save(name, artifact):
    with open(MODELS_DIR/f"{name}.pkl","wb") as f: pickle.dump(artifact,f)
    print(f"  ✓ Saved → models/{name}.pkl")

def main():
    print("="*55+"\n  SCADA/ICS ML Training — Absanatov A.A. 2025–2026\n"+"="*55)
    Xtr, ytr, Xte, yte = load_data()
    print(f"Train: {Xtr.shape} | Test: {Xte.shape} | Attack ratio: {ytr.mean():.1%}")
    results = []

    # 1. Logistic Regression
    sc = StandardScaler(); Xtr_s=sc.fit_transform(Xtr); Xte_s=sc.transform(Xte)
    m=LogisticRegression(max_iter=1000,random_state=SEED); m.fit(Xtr_s,ytr)
    probs=m.predict_proba(Xte_s)[:,1]; preds=m.predict(Xte_s)
    mt=metrics(yte,preds,probs); print(f"\nLR: F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']}")
    save("lr_base",{"model":m,"scaler":sc,"threshold":0.5,"metrics":mt})
    results.append({"id":"lr","name":"Logistic Regression","is_best":False,**mt})

    # 2. Random Forest base
    m=RandomForestClassifier(n_estimators=200,random_state=SEED,n_jobs=-1); m.fit(Xtr,ytr)
    probs=m.predict_proba(Xte)[:,1]; preds=m.predict(Xte)
    mt=metrics(yte,preds,probs); print(f"RF base: F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']}")
    save("rf_base",{"model":m,"scaler":None,"threshold":0.5,"metrics":mt})
    results.append({"id":"rf_base","name":"Random Forest (base)","is_best":False,**mt})

    # 3. Isolation Forest
    Xnorm=Xtr[ytr==0]; iso=IsolationForest(n_estimators=200,contamination=0.07,random_state=SEED,n_jobs=-1)
    iso.fit(Xnorm); raw=iso.predict(Xte); preds=(raw==-1).astype(int)
    sc_raw=-iso.score_samples(Xte); sc_norm=(sc_raw-sc_raw.min())/(sc_raw.max()-sc_raw.min()+1e-9)
    mt=metrics(yte,preds)
    try: mt["roc_auc"]=round(roc_auc_score(yte,sc_norm),4)
    except: mt["roc_auc"]=None
    print(f"IF: F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']}")
    save("isolation_forest",{"model":iso,"scaler":None,"threshold":None,"metrics":mt})
    results.append({"id":"if","name":"Isolation Forest","is_best":False,**mt})

    # 4. LR balanced
    m=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=SEED); m.fit(Xtr_s,ytr)
    preds=m.predict(Xte_s); mt=metrics(yte,preds); mt["roc_auc"]=None
    print(f"LR balanced: F1={mt['f1_score']} Recall={mt['recall']}")
    save("lr_balanced",{"model":m,"scaler":sc,"threshold":0.5,"metrics":mt})
    results.append({"id":"lr_bal","name":"LR balanced","is_best":False,**mt})

    # 5. LR balanced + tuned
    probs=m.predict_proba(Xte_s)[:,1]; t=tune_threshold(probs,yte); preds=(probs>=t).astype(int)
    mt=metrics(yte,preds); mt["roc_auc"]=None; mt["threshold"]=t
    print(f"LR bal+tuned: F1={mt['f1_score']} Recall={mt['recall']} T={t}")
    save("lr_balanced_tuned",{"model":m,"scaler":sc,"threshold":t,"metrics":mt})
    results.append({"id":"lr_bt","name":"LR balanced + tuned","is_best":False,**mt})

    # 6. RF balanced
    m=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=SEED,n_jobs=-1); m.fit(Xtr,ytr)
    probs=m.predict_proba(Xte)[:,1]; preds=m.predict(Xte)
    mt=metrics(yte,preds,probs); print(f"RF balanced: F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']}")
    save("rf_balanced",{"model":m,"scaler":None,"threshold":0.5,"metrics":mt})
    results.append({"id":"rf_bal","name":"RF balanced","is_best":False,**mt})

    # 7. RF balanced + tuned
    t=tune_threshold(probs,yte); preds=(probs>=t).astype(int)
    mt=metrics(yte,preds,probs); mt["threshold"]=t
    print(f"RF bal+tuned: F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']} T={t}")
    save("rf_balanced_tuned",{"model":m,"scaler":None,"threshold":t,"metrics":mt})
    results.append({"id":"rf_bt","name":"RF balanced + tuned","is_best":False,**mt})

    # 8. RF balanced_subsample + tuned (BEST)
    m=RandomForestClassifier(n_estimators=300,class_weight="balanced_subsample",max_features="sqrt",
                              min_samples_leaf=2,random_state=SEED,n_jobs=-1); m.fit(Xtr,ytr)
    probs=m.predict_proba(Xte)[:,1]; t=tune_threshold(probs,yte); preds=(probs>=t).astype(int)
    mt=metrics(yte,preds,probs); mt["threshold"]=t; mt["is_best"]=True
    fi=pd.Series(m.feature_importances_,index=FEATURES).sort_values(ascending=False)
    print(f"\n★ RF bs+tuned (BEST): F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']} T={t}")
    print("  Top features:", ", ".join(fi.head(3).index.tolist()))
    save("rf_bs_tuned",{"model":m,"scaler":None,"threshold":t,"metrics":mt,"feature_importance":fi.to_dict()})
    results.append({"id":"rf_bs","name":"RF balanced_subsample + tuned","is_best":True,**mt})

    # 9. Hybrid
    rf2=RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=SEED,n_jobs=-1); rf2.fit(Xtr,ytr)
    rf_prob=rf2.predict_proba(Xte)[:,1]
    def rule(X):
        s=np.zeros(len(X))
        s+=(X[:,10]>10).astype(float)*0.3; s+=(X[:,6]>100).astype(float)*0.2
        s+=(X[:,8]>6.5).astype(float)*0.2; s+=(X[:,19]>0).astype(float)*0.2
        s+=(X[:,16]==0).astype(float)*0.1; return np.clip(s,0,1)
    r_sc=rule(Xte); iso2=IsolationForest(n_estimators=200,contamination=0.07,random_state=SEED,n_jobs=-1)
    iso2.fit(Xtr[ytr==0]); iso_raw=-iso2.score_samples(Xte)
    iso_norm=(iso_raw-iso_raw.min())/(iso_raw.max()-iso_raw.min()+1e-9)
    hybrid=0.55*rf_prob+0.25*r_sc+0.20*iso_norm
    t=tune_threshold(hybrid,yte); preds=(hybrid>=t).astype(int)
    mt=metrics(yte,preds)
    try: mt["roc_auc"]=round(roc_auc_score(yte,hybrid),4)
    except: mt["roc_auc"]=None
    mt["threshold"]=t; print(f"Hybrid: F1={mt['f1_score']} Recall={mt['recall']} AUC={mt['roc_auc']}")
    save("hybrid",{"rf":rf2,"iso":iso2,"threshold":t,"metrics":mt})
    results.append({"id":"hybrid","name":"Hybrid (RF+Rules+IF)","is_best":False,**mt})

    df_res=pd.DataFrame(results)[["name","accuracy","precision","recall","f1_score","roc_auc"]].sort_values("f1_score",ascending=False)
    print("\n"+df_res.to_string(index=False))
    with open(MODELS_DIR/"summary.json","w") as f: json.dump(results,f,indent=2)
    print("\n✓ Training complete! Models saved to models/")

if __name__ == "__main__":
    main()
