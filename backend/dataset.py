"""
dataset.py — Синтетикалық SCADA/ICS трафик деректері генераторы
Газ тасымалдау кәсіпорнының желілік трафигін модельдейді
0 = қалыпты, 1 = шабуыл/аномалия
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

SEED = 42
rng  = np.random.default_rng(SEED)

FEATURES = [
    "pressure_bar", "flow_rate_m3h", "temperature_c", "valve_position_pct", "rpm_compressor",
    "pkt_size_bytes", "pkt_rate_pps", "inter_arrival_ms", "payload_entropy", "tcp_flags_numeric",
    "modbus_func_code", "dnp3_app_ctrl", "opc_tag_count", "cmd_response_delta_ms", "reg_write_freq",
    "hour_of_day", "src_ip_internal", "dst_port_industrial", "session_duration_s", "failed_auth_count",
]

def _gen_normal(n):
    return np.column_stack([
        rng.normal(60,5,n), rng.normal(800,60,n), rng.normal(15,3,n),
        rng.normal(50,8,n), rng.normal(1500,100,n), rng.normal(128,20,n),
        rng.normal(10,3,n), rng.normal(50,10,n), rng.uniform(3.5,4.5,n),
        rng.choice([2,16,18,24],n), rng.choice([1,3,4,6],n,p=[0.4,0.3,0.2,0.1]),
        rng.integers(0,64,n), rng.integers(1,10,n), rng.normal(5,2,n),
        rng.uniform(0.01,0.1,n), rng.integers(0,24,n),
        rng.choice([0,1],n,p=[0.05,0.95]), rng.choice([0,1],n,p=[0.02,0.98]),
        rng.normal(120,30,n), np.zeros(n),
    ])

def _gen_command_injection(n):
    d = _gen_normal(n)
    d[:,10] = rng.choice([5,15,22],n)
    d[:,4]  += rng.uniform(300,600,n)
    d[:,3]  = rng.uniform(0,5,n)
    d[:,14] += rng.uniform(0.5,2.0,n)
    d[:,13] = rng.normal(0.5,0.2,n)
    return d

def _gen_firmware_tampering(n):
    d = _gen_normal(n)
    d[:,6]  += rng.uniform(20,50,n)
    d[:,8]  += rng.uniform(1.5,3,n)
    d[:,7]  -= rng.uniform(10,30,n)
    d[:,1]  += rng.normal(50,20,n)
    return d

def _gen_replay_spoofing(n):
    d = _gen_normal(n)
    d[:,13] = rng.choice([0.1,200],n,p=[0.5,0.5])
    d[:,11] = rng.integers(64,128,n)
    d[:,6]  += rng.uniform(30,80,n)
    d[:,8]  = rng.uniform(6.5,8,n)
    return d

def _gen_mitm(n):
    d = _gen_normal(n)
    d[:,0]  = rng.normal(40,3,n)
    d[:,1]  = rng.normal(400,30,n)
    d[:,8]  = rng.uniform(5,7,n)
    d[:,9]  = rng.choice([1,9,17],n)
    d[:,16] = rng.choice([0,1],n,p=[0.5,0.5])
    return d

def _gen_dos(n):
    d = _gen_normal(n)
    d[:,6]  = rng.uniform(500,2000,n)
    d[:,5]  = rng.uniform(40,80,n)
    d[:,7]  = rng.uniform(0.1,1,n)
    d[:,16] = np.zeros(n)
    d[:,17] = np.zeros(n)
    return d

def _gen_eavesdropping(n):
    d = _gen_normal(n)
    d[:,5]  = rng.uniform(40,60,n)
    d[:,6]  = rng.uniform(1,3,n)
    d[:,12] = rng.integers(50,200,n)
    d[:,16] = np.zeros(n)
    return d

def _gen_ransomware(n):
    d = _gen_normal(n)
    d[:,8]  = rng.uniform(7,8,n)
    d[:,5]  = rng.uniform(1400,1500,n)
    d[:,6]  = rng.uniform(50,200,n)
    d[:,19] = rng.integers(1,10,n)
    d[:,18] = rng.uniform(1,30,n)
    return d

ATTACK_GENERATORS = [
    _gen_command_injection, _gen_firmware_tampering, _gen_replay_spoofing,
    _gen_mitm, _gen_dos, _gen_eavesdropping, _gen_ransomware,
]

def build_dataset(n_normal=5000, n_attacks_each=100, test_size=0.2, output_dir="data"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    X_normal = _gen_normal(n_normal)
    y_normal = np.zeros(n_normal, dtype=int)

    X_atk = np.vstack([g(n_attacks_each) for g in ATTACK_GENERATORS])
    y_atk  = np.ones(len(X_atk), dtype=int)

    X = np.vstack([X_normal, X_atk])
    y = np.concatenate([y_normal, y_atk])

    X[:,0]  = np.clip(X[:,0],  0, 120)
    X[:,1]  = np.clip(X[:,1],  0, 2000)
    X[:,3]  = np.clip(X[:,3],  0, 100)
    X[:,4]  = np.clip(X[:,4],  0, 3000)
    X[:,6]  = np.clip(X[:,6],  0, 2000)
    X[:,8]  = np.clip(X[:,8],  0, 8)
    X[:,15] = np.clip(X[:,15], 0, 23)

    df = pd.DataFrame(X, columns=FEATURES)
    df["label"] = y
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df["label"])
    df.to_csv(output_path / "scada_full.csv", index=False)
    train_df.to_csv(output_path / "scada_train.csv", index=False)
    test_df.to_csv(output_path / "scada_test.csv", index=False)

    print(f"Dataset: Total={len(df)} Train={len(train_df)} Test={len(test_df)}")
    print(f"Normal={( df.label==0).sum()} Attack={(df.label==1).sum()}")
    return df, train_df, test_df

if __name__ == "__main__":
    build_dataset()
