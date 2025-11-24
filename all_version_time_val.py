# -*- coding: utf-8 -*-
"""
主程式：f1/f2/f3 完成 CT 統計→累加→映射，並與 XGB/RF/Ada 做增量式融合評估
- 不覆蓋 ./dataset/4/train.csv
- 產出 CSV：artifacts/ 內含每輪評估與時間欄位（排除 I/O）

時間欄位：
1) 模型：rf/xgb/ada
   - *_fit_time_s：fit 計算時間
   - *_pred_test_time_s：test 集 predict_proba 計算時間 +（若本輪重訓）找 threshold 的時間
2) CT：
   - ct_stat_time_s：statistic 或 recompute 的純統計時間（f1/f2 回傳）
   - ct_map_time_s：f3 map_features_to_ct 的時間（val + test 兩次相加，不含讀表）
   - ct_result_time_s：找 threshold + 以 val MinMax 映射 test（confidence）的時間
   - ct_total_time_s：以上三者總和
"""

import os
import time
import shutil
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, f1_score, accuracy_score
)
from sklearn.preprocessing import MinMaxScaler

# === 依照你提供的檔案函式介面 ===
import ct_value.f1_statistic as f1   # statistic(df, out_dir='...') -> compute_secs（純統計時間）
import ct_value.f2_recompute as f2   # recompute(df, feature_count_dir='...') -> (tables, compute_secs)；會寫檔
import ct_value.f3_mapping as f3     # load_ratio_table(..., return_time=bool), map_features_to_ct(..., return_time=bool)

# -------------------- 路徑設定 --------------------
#存檔位置
ARTIFACT_DIR = "artifacts_ts_under"
FEATURE_COUNT_DIR = os.path.normpath(os.path.join(os.getcwd(), "feature_count"))

DATASET_CUR = os.path.join("dataset", "none")
TRAIN_CUR = os.path.join(DATASET_CUR, "train.csv")
VAL_CUR   = os.path.join(DATASET_CUR, "val.csv")   # ★ validation 直接讀檔
TEST_CUR  = os.path.join(DATASET_CUR, "test.csv")

# True是增量版， False是減量版
VERSION = True

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FEATURE_COUNT_DIR, exist_ok=True)
os.makedirs(DATASET_CUR, exist_ok=True)

# -------------------- 小工具 --------------------
def now():
    return time.perf_counter()  # 高解析度計時

def ensure_clean_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)

def find_optimal_threshold(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    if np.any(finite):
        j = tpr - fpr
        idxs = np.where(finite)[0]
        best = idxs[np.argmax(j[finite])]
        th = thresholds[best]
        if np.isfinite(th):
            return float(th)
    return float(np.median(y_score))

def evaluate_model(y_true, y_pred, y_score, model_usage=None):
    metrics = {
        'AUROC': roc_auc_score(y_true, y_score),
        'AUPRC': average_precision_score(y_true, y_score),
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        'TNR': tn / (tn + fp) if (tn + fp) else 0.0,
        'TPR': tp / (tp + fn) if (tp + fn) else 0.0,
        'PPV': tp / (tp + fp) if (tp + fp) else 0.0,
        'NPV': tn / (tn + fn) if (tn + fn) else 0.0
    })
    print("Confusion Matrix:")
    print(np.array([[tn, fp],[fn, tp]]))
    if model_usage:
        metrics.update(model_usage)
    return metrics

def count_model_selections(scores_list, names):
    abs_stack = np.vstack([np.abs(s) for s in scores_list])
    winners = np.argmax(abs_stack, axis=0)
    sel = {n: int(np.sum(winners == i)) for i, n in enumerate(names)}
    return sel

def minmax_confidence_from_val(val_scores, test_scores, thres):
    scaler = MinMaxScaler()
    scaler.fit(val_scores.reshape(-1, 1))
    test_scaled  = scaler.transform(test_scores.reshape(-1, 1))
    thres_scaled = scaler.transform(np.array([[thres]]))
    conf = (test_scaled - thres_scaled).ravel()
    return conf, scaler

def sanitize_matrix(df: pd.DataFrame, fill=0.0) -> pd.DataFrame:
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.fillna(fill)
    return out

def sanitize_vector(v: np.ndarray, fill=0.0) -> np.ndarray:
    v = np.asarray(v, dtype='float64')
    v = np.nan_to_num(v, nan=fill, posinf=fill, neginf=fill)
    return v

def compute_ratio_and_flag(y: np.ndarray):
    """
    CT table 用的 black_more / ratio（跟採樣無關，保留原本邏輯）
    """
    n_black = int((y == 1).sum())
    n_white = int((y == 0).sum())
    if n_black >= n_white:
        black_more = True
        if VERSION:
            ratio = n_black / max(n_white, 1)
        else:
            ratio = n_white / max(n_black, 1)
    else:
        black_more = False
        if VERSION:
            ratio = n_white / max(n_black, 1)
        else:
            ratio = n_black / max(n_white, 1)
    return ratio, black_more

def Get_Scale_Pos_Weight(sensitivity):
    return sensitivity * 30

# === 新增：根據「驗證集比例」決定採樣數量 ===
def get_increment_size(current_size: int) -> int:
    """
    根據當前訓練資料量決定每輪要加入的「總筆數」：
    - N < 10,000       → 1000
    - 10,000 ≤ N < 100,000  → 1000
    - 100,000 ≤ N < 1,000,000 → 10,000
    - 之後             → 100,000
    """
    if current_size < 10_000:
        return 1000
    elif current_size < 100_000:
        return 1000
    elif current_size < 1_000_000:
        return 10_000
    else:
        return 100_000

def compute_neg_pos_for_total(total_inc: int,
                              neg_pos_ratio: float,
                              max_neg: int,
                              max_pos: int):
    """
    total_inc: 本輪希望加入的總筆數（例如 1000）
    neg_pos_ratio: 驗證集中的 neg/pos 比例 (neg:pos)
    max_neg / max_pos: 目前剩餘池中可用的數量（不能超過）
    回傳 (n_neg, n_pos)
    """
    if total_inc <= 0 or max_neg + max_pos <= 0:
        return 0, 0

    # 依 neg:pos = R:1 反推 pos 數量
    pos = int(round(total_inc / (neg_pos_ratio + 1.0))) if neg_pos_ratio > 0 else int(total_inc / 2)
    pos = max(1, pos)
    neg = total_inc - pos

    # 不可超過池中上限
    pos = min(pos, max_pos)
    neg = min(neg, max_neg)

    # 若這樣導致 total < 2，或其中一類為 0，後面會再處理
    return neg, pos

# ---- 每輪都重訓的 XGB 對照組 ----
xgb_comparison_results = []
def run_xgb_comp(round_id, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
                 train_fit_df, artifacts_dir):
    xgb_comp = XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=100,
        scale_pos_weight=Get_Scale_Pos_Weight(sensitivity=0.85),
        use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=None
    )

    t0 = now(); xgb_comp.fit(X_train_fit, y_train_fit); fit_dt = now() - t0

    val_scores = sanitize_vector(xgb_comp.predict_proba(X_val)[:, 1])
    t0 = now(); th = find_optimal_threshold(y_val, val_scores); th_dt = now() - t0

    t0 = now(); test_scores = sanitize_vector(xgb_comp.predict_proba(X_test)[:, 1]); pred_dt = now() - t0
    pred_test_time_s = th_dt + pred_dt

    y_pred = (test_scores > th).astype(int)
    m = evaluate_model(y_test, y_pred, test_scores, {'model': 'XGB_comp'})

    row = {
        'Round': round_id,
        'train_0': int((train_fit_df['label'] == 0).sum()),
        'train_1': int((train_fit_df['label'] == 1).sum()),
        'threshold': th,
        'fit_time_s': fit_dt,
        'pred_test_time_s': pred_test_time_s,
        'AUROC': m['AUROC'], 'AUPRC': m['AUPRC'], 'Accuracy': m['Accuracy'], 'F1': m['F1'],
        'TNR': m['TNR'], 'TPR': m['TPR'], 'PPV': m['PPV'], 'NPV': m['NPV'],
    }
    xgb_comparison_results.append(row)
    pd.DataFrame(xgb_comparison_results).to_csv(
        os.path.join(artifacts_dir, 'xgb_comparison_results.csv'), index=False
    )

# -------------------- 結果累積器 --------------------
ct_only_results = []
ct_xg_results = []
ct_xg_rf_results = []
ct_xg_rf_ada_results = []
xg_only_results = []         # XGB
xg_rf_results = []           # XGB + RF
xg_rf_ada_results = []       # XGB + RF + ADA

last_ct_scaler = None

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    # 讀資料
    train_df = pd.read_csv(TRAIN_CUR, low_memory=False)
    val_df   = pd.read_csv(VAL_CUR,   low_memory=False)  # ★ validation
    test_df  = pd.read_csv(TEST_CUR,  low_memory=False)

    # --- 根據「驗證集」計算 neg:pos 比例，供後續所有採樣使用 ---
    val_labels = val_df['label'].values
    val_pos = int((val_labels == 1).sum())
    val_neg = int((val_labels == 0).sum())
    if val_pos == 0 or val_neg == 0:
        print("[WARN] Validation set 極端不平衡，採用預設 neg:pos = 30:1")
        val_neg_pos_ratio = 30.0
    else:
        val_neg_pos_ratio = val_neg / val_pos
    print(f"[INFO] Validation neg:pos ratio ≈ {val_neg_pos_ratio:.3f} (neg/pos)")

    # train 快照（一次）
    RAW_TRAIN_SNAPSHOT = os.path.join(ARTIFACT_DIR, "raw_train_snapshot.csv")
    if not os.path.exists(RAW_TRAIN_SNAPSHOT):
        try:
            shutil.copy2(TRAIN_CUR, RAW_TRAIN_SNAPSHOT)
        except Exception as e:
            print(f"[WARN] 建立原始 train 快照失敗：{e}")

    # ------------------ 初始採樣：總數 1000，比例依驗證集 ------------------
    df0 = train_df[train_df['label'] == 0]
    df1 = train_df[train_df['label'] == 1]

    total_init = min(1000, len(train_df))
    init_neg, init_pos = compute_neg_pos_for_total(
        total_inc=total_init,
        neg_pos_ratio=val_neg_pos_ratio,
        max_neg=len(df0),
        max_pos=len(df1)
    )

    # 若其中一類為 0，做一點補救，至少各取 1（若資料夠）
    if init_neg == 0 and len(df0) > 0:
        init_neg = min(len(df0), total_init - 1)
    if init_pos == 0 and len(df1) > 0:
        init_pos = min(len(df1), total_init - init_neg)

    print(f"[INFO] Initial seed total={init_neg+init_pos}, neg={init_neg}, pos={init_pos}")

    seed0 = df0.sample(n=init_neg, random_state=7709)
    seed1 = df1.sample(n=init_pos, random_state=7709)
    train_fit_df = pd.concat([seed0, seed1]).sample(frac=1.0, random_state=7709).copy()

    # 剩餘樣本池（供後續增量）：所有不在 seed 裡的 train
    remainder_pool = train_df.drop(index=train_fit_df.index)

    # X / y
    X_train_fit = sanitize_matrix(train_fit_df.drop(columns=['label']))
    y_train_fit = train_fit_df['label'].values
    X_val = sanitize_matrix(val_df.drop(columns=['label']))   # ★ 來自 val.csv
    y_val = val_df['label'].values
    X_test = sanitize_matrix(test_df.drop(columns=['label']))
    y_test = test_df['label'].values

    # 初始化模型與訓練（fit 計時；固定 n_jobs=None 使資源一致）
    rf  = RandomForestClassifier(
        n_estimators=150, class_weight="balanced",
        random_state=42, max_depth=10, n_jobs=None
    )
    xgb = XGBClassifier(
        max_depth=6, learning_rate=0.1, n_estimators=100,
        scale_pos_weight=Get_Scale_Pos_Weight(sensitivity=0.85),
        use_label_encoder=False, eval_metric='logloss',
        random_state=42, n_jobs=None
    )
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)

    t0 = now(); rf.fit(X_train_fit, y_train_fit);   rf_fit_time_s  = now()-t0
    t0 = now(); xgb.fit(X_train_fit, y_train_fit);  xg_fit_time_s  = now()-t0
    t0 = now(); ada.fit(X_train_fit, y_train_fit);  ada_fit_time_s = now()-t0

    # ---- 先在 val 上求各自門檻（時間稍後合併到 pred_test_time_s）----
    rf_val  = sanitize_vector(rf.predict_proba(X_val)[:, 1])
    xg_val  = sanitize_vector(xgb.predict_proba(X_val)[:, 1])
    ada_val = sanitize_vector(ada.predict_proba(X_val)[:, 1])

    t0 = now(); rf_th  = find_optimal_threshold(y_val, rf_val);   rf_th_time  = now()-t0
    t0 = now(); xg_th  = find_optimal_threshold(y_val, xg_val);   xg_th_time  = now()-t0
    t0 = now(); ada_th = find_optimal_threshold(y_val, ada_val);  ada_th_time = now()-t0

    thresholds = {'rf': rf_th, 'xg': xg_th, 'ada': ada_th}
    with open(os.path.join(ARTIFACT_DIR, 'thresholds_round0.pkl'), 'wb') as f:
        pickle.dump(thresholds, f)

    # ===== 第一次 CT：statistic + mapping + result（全都排除 I/O）=====
    ensure_clean_dir(FEATURE_COUNT_DIR)

    # f1：純統計時間
    ct_stat_time_s = f1.statistic(train_fit_df.copy(), out_dir=FEATURE_COUNT_DIR)

    # f3：載入 ratio table → 映射
    ratio_ct, black_more = compute_ratio_and_flag(y_train_fit)
    ct_table, _ = f3.load_ratio_table(FEATURE_COUNT_DIR, ratio_ct, black_more, return_time=True, version=VERSION)

    ct_val_df,  map_val_secs  = f3.map_features_to_ct(X_val.copy(),  ct_table, return_time=True)
    ct_test_df, map_test_secs = f3.map_features_to_ct(X_test.copy(), ct_table, return_time=True)
    ct_map_time_s = map_val_secs + map_test_secs

    ct_val_raw  = -ct_val_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values
    ct_test_raw = -ct_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values

    t0 = now()
    ct_th = find_optimal_threshold(y_val, ct_val_raw)
    ct_conf_test, ct_scaler = minmax_confidence_from_val(ct_val_raw, ct_test_raw, ct_th)
    ct_result_time_s = now() - t0
    ct_total_time_s  = ct_stat_time_s + ct_map_time_s + ct_result_time_s
    last_ct_scaler = ct_scaler

    # 模型：test 預測（+ 門檻時間）
    t0 = now(); rf_test  = sanitize_vector(rf.predict_proba(X_test)[:, 1]);  rf_pred_test_time_s  = (now()-t0) + rf_th_time
    t0 = now(); xg_test  = sanitize_vector(xgb.predict_proba(X_test)[:, 1]); xg_pred_test_time_s = (now()-t0) + xg_th_time
    t0 = now(); ada_test = sanitize_vector(ada.predict_proba(X_test)[:, 1]); ada_pred_test_time_s = (now()-t0) + ada_th_time

    rf_conf_test,  rf_scaler  = minmax_confidence_from_val(rf_val,  rf_test,  thresholds['rf'])
    xg_conf_test,  xg_scaler  = minmax_confidence_from_val(xg_val,  xg_test,  thresholds['xg'])
    ada_conf_test, ada_scaler = minmax_confidence_from_val(ada_val, ada_test, thresholds['ada'])

    with open(os.path.join(ARTIFACT_DIR, "rf_scaler.pkl"), "wb") as f:  pickle.dump(rf_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "xg_scaler.pkl"), "wb") as f:  pickle.dump(xg_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "ada_scaler.pkl"), "wb") as f: pickle.dump(ada_scaler, f)

    # ---- 第 0 輪寫入 ----
    def log_round(round_id):
        common = {
            'Round': round_id,
            'train_0': int((train_fit_df['label'] == 0).sum()),
            'train_1': int((train_fit_df['label'] == 1).sum()),
            'ct_thres': ct_th, 'rf_thres': thresholds['rf'], 'xg_thres': thresholds['xg'], 'ada_thres': thresholds['ada'],
            'refresh': (round_id % 10 == 0),
            # --- 時間欄位 ---
            'ct_stat_time_s': ct_stat_time_s,
            'ct_map_time_s': ct_map_time_s,
            'ct_result_time_s': ct_result_time_s,
            'ct_total_time_s': ct_total_time_s,
            'rf_fit_time_s': rf_fit_time_s, 'xg_fit_time_s': xg_fit_time_s, 'ada_fit_time_s': ada_fit_time_s,
            'rf_pred_test_time_s': rf_pred_test_time_s, 'xg_pred_test_time_s': xg_pred_test_time_s, 'ada_pred_test_time_s': ada_pred_test_time_s,
        }

        # XGB only
        xg_bin = (xg_conf_test > 0).astype(int)
        mxg = evaluate_model(y_test, xg_bin, xg_conf_test, {'model': 'XGB'})
        xg_only_results.append({**common, **mxg})

        # XGB + RF
        pick_xr = np.where(np.abs(xg_conf_test) >= np.abs(rf_conf_test), xg_conf_test, rf_conf_test)
        bin_xr  = (pick_xr > 0).astype(int)
        sel_xr  = count_model_selections([xg_conf_test, rf_conf_test], names=['xg','rf'])
        mxr = evaluate_model(y_test, bin_xr, pick_xr, sel_xr)
        xg_rf_results.append({**common, **mxr})

        # XGB + RF + ADA
        pick_xra = np.where(
            (np.abs(xg_conf_test) >= np.abs(rf_conf_test)) & (np.abs(xg_conf_test) >= np.abs(ada_conf_test)),
            xg_conf_test,
            np.where(np.abs(rf_conf_test) >= np.abs(ada_conf_test), rf_conf_test, ada_conf_test)
        )
        bin_xra = (pick_xra > 0).astype(int)
        sel_xra = count_model_selections([xg_conf_test, rf_conf_test, ada_conf_test], names=['xg','rf','ada'])
        mxra = evaluate_model(y_test, bin_xra, pick_xra, sel_xra)
        xg_rf_ada_results.append({**common, **mxra})

        # CT only
        ct_bin = (ct_conf_test > 0).astype(int)
        m_ct = evaluate_model(y_test, ct_bin, ct_conf_test, {'ct_selected': len(y_test)})
        ct_only_results.append({**common, **m_ct})

        # CT + XGB
        pick_cx = np.where(np.abs(ct_conf_test) >= np.abs(xg_conf_test), ct_conf_test, xg_conf_test)
        bin_cx  = (pick_cx > 0).astype(int)
        sel_cx  = count_model_selections([ct_conf_test, xg_conf_test], names=['ct','xg'])
        m_cx = evaluate_model(y_test, bin_cx, pick_cx, sel_cx)
        ct_xg_results.append({**common, **m_cx})

        # CT + XGB + RF
        pick_cxr = np.where(
            (np.abs(ct_conf_test) >= np.abs(xg_conf_test)) & (np.abs(ct_conf_test) >= np.abs(rf_conf_test)),
            ct_conf_test,
            np.where(np.abs(xg_conf_test) >= np.abs(rf_conf_test), xg_conf_test, rf_conf_test)
        )
        bin_cxr = (pick_cxr > 0).astype(int)
        sel_cxr = count_model_selections([ct_conf_test, xg_conf_test, rf_conf_test], names=['ct','xg','rf'])
        m_cxr = evaluate_model(y_test, bin_cxr, pick_cxr, sel_cxr)
        ct_xg_rf_results.append({**common, **m_cxr})

        # CT + XGB + RF + ADA
        pick_full = np.where(
            (np.abs(ct_conf_test) >= np.abs(xg_conf_test)) &
            (np.abs(ct_conf_test) >= np.abs(rf_conf_test)) &
            (np.abs(ct_conf_test) >= np.abs(ada_conf_test)),
            ct_conf_test,
            np.where(
                (np.abs(xg_conf_test) >= np.abs(rf_conf_test)) & (np.abs(xg_conf_test) >= np.abs(ada_conf_test)),
                xg_conf_test,
                np.where(np.abs(rf_conf_test) >= np.abs(ada_conf_test), rf_conf_test, ada_conf_test)
            )
        )
        bin_full = (pick_full > 0).astype(int)
        sel_full = count_model_selections(
            [ct_conf_test, xg_conf_test, rf_conf_test, ada_conf_test],
            names=['ct','xg','rf','ada']
        )
        m_full = evaluate_model(y_test, bin_full, pick_full, sel_full)
        ct_xg_rf_ada_results.append({**common, **m_full})

        # 寫檔
        pd.DataFrame(xg_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'xg_only_results.csv'), index=False)
        pd.DataFrame(xg_rf_results).to_csv(os.path.join(ARTIFACT_DIR, 'xg_rf_results.csv'), index=False)
        pd.DataFrame(xg_rf_ada_results).to_csv(os.path.join(ARTIFACT_DIR, 'xg_rf_ada_results.csv'), index=False)
        pd.DataFrame(ct_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_only_results.csv'), index=False)
        pd.DataFrame(ct_xg_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_xg_results.csv'), index=False)
        pd.DataFrame(ct_xg_rf_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_xg_rf_results.csv'), index=False)
        pd.DataFrame(ct_xg_rf_ada_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_xg_rf_ada_results.csv'), index=False)

    # 第 0 輪
    log_round(0)
    # XGB 對照組：第 0 輪也重訓一次
    run_xgb_comp(
        0, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
        train_fit_df, ARTIFACT_DIR
    )

    # ===== 增量迭代（依「驗證集比例」動態增量，直到資料用完）=====
    remaining = remainder_pool.copy()
    rem_0 = remaining[remaining['label'] == 0].copy()
    rem_1 = remaining[remaining['label'] == 1].copy()

    round_id = 1
    while len(rem_0) > 0 and len(rem_1) > 0:
        current_size = len(train_fit_df)
        total_inc = get_increment_size(current_size)

        # 不可超過剩餘資料總數
        remaining_total = len(rem_0) + len(rem_1)
        total_inc = min(total_inc, remaining_total)
        if total_inc <= 0:
            break

        # 依驗證集 neg:pos 比例拆成本輪欲加入的 neg / pos 數
        desired_neg, desired_pos = compute_neg_pos_for_total(
            total_inc=total_inc,
            neg_pos_ratio=val_neg_pos_ratio,
            max_neg=len(rem_0),
            max_pos=len(rem_1)
        )

        # 如果算出來有一類是 0，但池中仍有資料，做最後一點補救
        if (desired_neg == 0 or desired_pos == 0):
            if len(rem_0) == 0 or len(rem_1) == 0:
                # 其中一類真的沒資料了，就結束
                break
            # 強制至少各 1 筆（若剩餘足夠）
            desired_pos = min(len(rem_1), max(1, int(round(total_inc / (val_neg_pos_ratio + 1.0)))))
            desired_neg = min(len(rem_0), total_inc - desired_pos)
            if desired_neg <= 0 or desired_pos <= 0:
                # 還是出問題就直接跳出
                break

        # 從池中抽樣
        batch_pos = rem_1.sample(n=desired_pos, random_state=round_id)
        batch_neg = rem_0.sample(n=desired_neg, random_state=round_id)
        new_batch = pd.concat([batch_neg, batch_pos]).sample(frac=1.0, random_state=round_id)

        print(f"[Round {round_id}] +{len(new_batch)} (neg={desired_neg}, pos={desired_pos}), current total = {current_size + len(new_batch)}")

        # 併入訓練集
        train_fit_df = pd.concat([train_fit_df, new_batch], ignore_index=True)
        X_train_fit = sanitize_matrix(train_fit_df.drop(columns=['label']))
        y_train_fit = train_fit_df['label'].values

        # f2：統計累加
        _, ct_stat_time_s = f2.recompute(train_fit_df, feature_count_dir=FEATURE_COUNT_DIR)

        # 最新比例 → 載表 → 映射
        ratio_ct, black_more = compute_ratio_and_flag(y_train_fit)
        ct_table, _ = f3.load_ratio_table(FEATURE_COUNT_DIR, ratio_ct, black_more, return_time=True,  version=VERSION)

        ct_val_df,  map_val_secs  = f3.map_features_to_ct(X_val.copy(),  ct_table, return_time=True)
        ct_test_df, map_test_secs = f3.map_features_to_ct(X_test.copy(), ct_table, return_time=True)
        ct_map_time_s = map_val_secs + map_test_secs

        ct_val_raw  = -ct_val_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values
        ct_test_raw = -ct_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values

        t0 = now()
        ct_th = find_optimal_threshold(y_val, ct_val_raw)
        ct_conf_test, ct_scaler = minmax_confidence_from_val(ct_val_raw, ct_test_raw, ct_th)
        ct_result_time_s = now() - t0
        ct_total_time_s  = ct_stat_time_s + ct_map_time_s + ct_result_time_s
        last_ct_scaler = ct_scaler

        # 每 10 輪重訓（量測 fit），並重新找門檻
        rf_fit_time_s = xg_fit_time_s = ada_fit_time_s = 0.0
        rf_th_time = xg_th_time = ada_th_time = 0.0
        if round_id % 10 == 0:
            t0 = now(); rf.fit(X_train_fit, y_train_fit);  rf_fit_time_s = now()-t0
            t0 = now(); xgb.fit(X_train_fit, y_train_fit); xg_fit_time_s = now()-t0
            t0 = now(); ada.fit(X_train_fit, y_train_fit); ada_fit_time_s = now()-t0

            rf_val  = sanitize_vector(rf.predict_proba(X_val)[:, 1])
            xg_val  = sanitize_vector(xgb.predict_proba(X_val)[:, 1])
            ada_val = sanitize_vector(ada.predict_proba(X_val)[:, 1])

            t0 = now(); thresholds['rf']  = find_optimal_threshold(y_val, rf_val);  rf_th_time  = now()-t0
            t0 = now(); thresholds['xg']  = find_optimal_threshold(y_val, xg_val);  xg_th_time  = now()-t0
            t0 = now(); thresholds['ada'] = find_optimal_threshold(y_val, ada_val); ada_th_time = now()-t0

            with open(os.path.join(ARTIFACT_DIR, f'thresholds_round{round_id}.pkl'), 'wb') as f:
                pickle.dump(thresholds, f)

        # 測試集預測
        rf_val  = sanitize_vector(rf.predict_proba(X_val)[:, 1])
        t0 = now(); rf_test  = sanitize_vector(rf.predict_proba(X_test)[:, 1]);  rf_pred_test_time_s  = (now()-t0) + rf_th_time
        rf_conf_test, rf_scaler = minmax_confidence_from_val(rf_val, rf_test, thresholds['rf'])

        xg_val  = sanitize_vector(xgb.predict_proba(X_val)[:, 1])
        t0 = now(); xg_test  = sanitize_vector(xgb.predict_proba(X_test)[:, 1]); xg_pred_test_time_s = (now()-t0) + xg_th_time
        xg_conf_test, xg_scaler = minmax_confidence_from_val(xg_val, xg_test, thresholds['xg'])

        ada_val = sanitize_vector(ada.predict_proba(X_val)[:, 1])
        t0 = now(); ada_test = sanitize_vector(ada.predict_proba(X_test)[:, 1]);  ada_pred_test_time_s = (now()-t0) + ada_th_time
        ada_conf_test, ada_scaler = minmax_confidence_from_val(ada_val, ada_test, thresholds['ada'])

        # ---- 寫一輪 ----
        def log_round_k():
            common = {
                'Round': round_id,
                'train_0': int((train_fit_df['label'] == 0).sum()),
                'train_1': int((train_fit_df['label'] == 1).sum()),
                'ct_thres': ct_th, 'rf_thres': thresholds['rf'], 'xg_thres': thresholds['xg'], 'ada_thres': thresholds['ada'],
                'refresh': (round_id % 10 == 0),
                # --- 時間欄位 ---
                'ct_stat_time_s': ct_stat_time_s,
                'ct_map_time_s': ct_map_time_s,
                'ct_result_time_s': ct_result_time_s,
                'ct_total_time_s': ct_total_time_s,
                'rf_fit_time_s': rf_fit_time_s, 'xg_fit_time_s': xg_fit_time_s, 'ada_fit_time_s': ada_fit_time_s,
                'rf_pred_test_time_s': rf_pred_test_time_s, 'xg_pred_test_time_s': xg_pred_test_time_s, 'ada_pred_test_time_s': ada_pred_test_time_s,
            }

            # XGB only
            xg_bin = (xg_conf_test > 0).astype(int)
            mxg = evaluate_model(y_test, xg_bin, xg_conf_test, {'model': 'XGB'})
            xg_only_results.append({**common, **mxg})

            # XGB + RF
            pick_xr = np.where(np.abs(xg_conf_test) >= np.abs(rf_conf_test), xg_conf_test, rf_conf_test)
            bin_xr  = (pick_xr > 0).astype(int)
            sel_xr  = count_model_selections([xg_conf_test, rf_conf_test], names=['xg','rf'])
            mxr = evaluate_model(y_test, bin_xr, pick_xr, sel_xr)
            xg_rf_results.append({**common, **mxr})

            # XGB + RF + ADA
            pick_xra = np.where(
                (np.abs(xg_conf_test) >= np.abs(rf_conf_test)) & (np.abs(xg_conf_test) >= np.abs(ada_conf_test)),
                xg_conf_test,
                np.where(np.abs(rf_conf_test) >= np.abs(ada_conf_test), rf_conf_test, ada_conf_test)
            )
            bin_xra = (pick_xra > 0).astype(int)
            sel_xra = count_model_selections([xg_conf_test, rf_conf_test, ada_conf_test], names=['xg','rf','ada'])
            mxra = evaluate_model(y_test, bin_xra, pick_xra, sel_xra)
            xg_rf_ada_results.append({**common, **mxra})

            # CT only
            ct_bin = (ct_conf_test > 0).astype(int)
            m_ct = evaluate_model(y_test, ct_bin, ct_conf_test, {'ct_selected': len(y_test)})
            ct_only_results.append({**common, **m_ct})

            # CT + XGB
            pick_cx = np.where(np.abs(ct_conf_test) >= np.abs(xg_conf_test), ct_conf_test, xg_conf_test)
            bin_cx  = (pick_cx > 0).astype(int)
            sel_cx  = count_model_selections([ct_conf_test, xg_conf_test], names=['ct','xg'])
            m_cx = evaluate_model(y_test, bin_cx, pick_cx, sel_cx)
            ct_xg_results.append({**common, **m_cx})

            # CT + XGB + RF
            pick_cxr = np.where(
                (np.abs(ct_conf_test) >= np.abs(xg_conf_test)) & (np.abs(ct_conf_test) >= np.abs(rf_conf_test)),
                ct_conf_test,
                np.where(np.abs(xg_conf_test) >= np.abs(rf_conf_test), xg_conf_test, rf_conf_test)
            )
            bin_cxr = (pick_cxr > 0).astype(int)
            sel_cxr = count_model_selections([ct_conf_test, xg_conf_test, rf_conf_test], names=['ct','xg','rf'])
            m_cxr = evaluate_model(y_test, bin_cxr, pick_cxr, sel_cxr)
            ct_xg_rf_results.append({**common, **m_cxr})

            # CT + XGB + RF + ADA
            pick_full = np.where(
                (np.abs(ct_conf_test) >= np.abs(xg_conf_test)) &
                (np.abs(ct_conf_test) >= np.abs(rf_conf_test)) &
                (np.abs(ct_conf_test) >= np.abs(ada_conf_test)),
                ct_conf_test,
                np.where(
                    (np.abs(xg_conf_test) >= np.abs(rf_conf_test)) & (np.abs(xg_conf_test) >= np.abs(ada_conf_test)),
                    xg_conf_test,
                    np.where(np.abs(rf_conf_test) >= np.abs(ada_conf_test), rf_conf_test, ada_conf_test)
                )
            )
            bin_full = (pick_full > 0).astype(int)
            sel_full = count_model_selections(
                [ct_conf_test, xg_conf_test, rf_conf_test, ada_conf_test],
                names=['ct','xg','rf','ada']
            )
            m_full = evaluate_model(y_test, bin_full, pick_full, sel_full)
            ct_xg_rf_ada_results.append({**common, **m_full})

            # 寫檔
            pd.DataFrame(xg_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'xg_only_results.csv'), index=False)
            pd.DataFrame(xg_rf_results).to_csv(os.path.join(ARTIFACT_DIR, 'xg_rf_results.csv'), index=False)
            pd.DataFrame(xg_rf_ada_results).to_csv(os.path.join(ARTIFACT_DIR, 'xg_rf_ada_results.csv'), index=False)
            pd.DataFrame(ct_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_only_results.csv'), index=False)
            pd.DataFrame(ct_xg_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_xg_results.csv'), index=False)
            pd.DataFrame(ct_xg_rf_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_xg_rf_results.csv'), index=False)
            pd.DataFrame(ct_xg_rf_ada_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_xg_rf_ada_results.csv'), index=False)

        log_round_k()

        # XGB 對照組：每一輪都重訓
        run_xgb_comp(
            round_id, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
            train_fit_df, ARTIFACT_DIR
        )

        # 從池刪除本批
        rem_0 = rem_0.drop(batch_neg.index, errors='ignore')
        rem_1 = rem_1.drop(batch_pos.index, errors='ignore')

        round_id += 1

    # 存最後一個 CT scaler
    if last_ct_scaler is not None:
        with open(os.path.join(ARTIFACT_DIR, "ct_scaler.pkl"), "wb") as f:
            pickle.dump(last_ct_scaler, f)

    print("程式執行完成")
