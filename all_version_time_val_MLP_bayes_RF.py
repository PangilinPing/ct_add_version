# -*- coding: utf-8 -*-
"""
主程式：f1/f2/f3 完成 CT 統計→累加→映射，並做以下評估：
1) 組合模型：CT + RF
2) 比較模型：Bayes、MLP

更新規則：
- Bayes：每輪更新
- MLP：每輪更新
- CT + RF 裡的 RF：每 10 輪更新一次

時間欄位：
1) 模型：rf / bayes / mlp
   - rf_fit_time_s：fit 計算時間（只有 round % 10 == 0 時才會非 0）
   - rf_pred_test_time_s：test 集 predict_proba 計算時間 +（若本輪重訓）找 threshold 的時間
   - bayes_fit_time_s：fit 計算時間（每輪）
   - bayes_pred_test_time_s：test 集 predict_proba 計算時間 + 找 threshold 的時間（每輪）
   - mlp_fit_time_s：fit 計算時間（每輪）
   - mlp_pred_test_time_s：test 集 predict_proba 計算時間 + 找 threshold 的時間（每輪）
2) CT：
   - ct_stat_time_s：statistic 或 recompute 的純統計時間（f1/f2 回傳）
   - ct_map_time_s：f3 map_features_to_ct 的時間（val + test 兩次相加，不含讀表）
   - ct_result_time_s：找 threshold + 以 val MinMax 映射 test（confidence）的時間
   - ct_total_time_s：以上三者總和

輸出：
- ct_rf_results.csv：CT + RF
- bayes_comparison_results.csv：每輪都重訓 Bayes
- mlp_comparison_results.csv：每輪都重訓 MLP
- rf_only_results.csv：RF only（RF 每10輪更新一次）
- ct_only_results.csv：CT only
"""

import os
import time
import shutil
import pickle
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, f1_score, accuracy_score
)
from sklearn.preprocessing import MinMaxScaler

# === 依照你提供的檔案函式介面 ===
import ct_value.f1_statistic as f1
import ct_value.f2_recompute as f2
import ct_value.f3_mapping as f3


# -------------------- 路徑設定 --------------------
ARTIFACT_DIR = "NB15_ct_rf_vs_bayes_mlp_resultv2"
FEATURE_COUNT_DIR = os.path.normpath(os.path.join(os.getcwd(), "feature_count"))

DATASET_CUR = os.path.join("dataset", "NB15")
TRAIN_CUR = os.path.join(DATASET_CUR, "train.csv")
VAL_CUR   = os.path.join(DATASET_CUR, "valid.csv")
TEST_CUR  = os.path.join(DATASET_CUR, "test.csv")

# True = 開啟 CT 平衡
# False = 關閉 CT 平衡
CT_BALANCE_ENABLED = True

os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(FEATURE_COUNT_DIR, exist_ok=True)
os.makedirs(DATASET_CUR, exist_ok=True)


# -------------------- 小工具 --------------------
def now():
    return time.perf_counter()


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
    print(np.array([[tn, fp], [fn, tp]]))
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
    test_scaled = scaler.transform(test_scores.reshape(-1, 1))
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


def compute_ratio_and_flag(y: np.ndarray, ct_balance_enabled: bool = True):
    """
    CT table 用的 black_more / ratio（跟採樣無關，保留原本邏輯）
    """
    n_black = int((y == 1).sum())
    n_white = int((y == 0).sum())

    if n_black >= n_white:
        black_more = True
        if ct_balance_enabled:
            ratio = n_black / max(n_white, 1)
        else:
            ratio = n_white / max(n_black, 1)
    else:
        black_more = False
        if ct_balance_enabled:
            ratio = n_white / max(n_black, 1)
        else:
            ratio = n_black / max(n_white, 1)

    return ratio, black_more


def get_increment_size(current_size: int) -> int:
    if current_size < 10_000:
        return 100
    elif current_size < 100_000:
        return 1000
    elif current_size < 1_000_000:
        return 10_000
    else:
        return 100_000


def compute_neg_pos_for_total(total_inc: int, neg_pos_ratio: float, max_neg: int, max_pos: int):
    if total_inc <= 0 or max_neg + max_pos <= 0:
        return 0, 0

    pos = int(round(total_inc / (neg_pos_ratio + 1.0))) if neg_pos_ratio > 0 else int(total_inc / 2)
    pos = max(1, pos)
    neg = total_inc - pos

    pos = min(pos, max_pos)
    neg = min(neg, max_neg)
    return neg, pos


# -------------------- comparison 模型結果 --------------------
bayes_comparison_results = []
mlp_comparison_results = []


def build_rf():
    return RandomForestClassifier(
        n_estimators=150,
        class_weight="balanced",
        random_state=42,
        max_depth=10,
        n_jobs=None
    )


def build_bayes():
    return GaussianNB()


def build_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=False,
        verbose=False
    )


def run_bayes_comp(round_id, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
                   train_fit_df, artifacts_dir):
    bayes_comp = build_bayes()

    t0 = now()
    bayes_comp.fit(X_train_fit, y_train_fit)
    fit_dt = now() - t0

    val_scores = sanitize_vector(bayes_comp.predict_proba(X_val)[:, 1])

    t0 = now()
    th = find_optimal_threshold(y_val, val_scores)
    th_dt = now() - t0

    t0 = now()
    test_scores = sanitize_vector(bayes_comp.predict_proba(X_test)[:, 1])
    pred_dt = now() - t0

    pred_test_time_s = th_dt + pred_dt
    y_pred = (test_scores > th).astype(int)

    m = evaluate_model(y_test, y_pred, test_scores, {'model': 'Bayes_comp'})

    row = {
        'Round': round_id,
        'train_0': int((train_fit_df['label'] == 0).sum()),
        'train_1': int((train_fit_df['label'] == 1).sum()),
        'threshold': th,
        'fit_time_s': fit_dt,
        'pred_test_time_s': pred_test_time_s,
        'AUROC': m['AUROC'],
        'AUPRC': m['AUPRC'],
        'Accuracy': m['Accuracy'],
        'F1': m['F1'],
        'TNR': m['TNR'],
        'TPR': m['TPR'],
        'PPV': m['PPV'],
        'NPV': m['NPV'],
    }
    bayes_comparison_results.append(row)
    pd.DataFrame(bayes_comparison_results).to_csv(
        os.path.join(artifacts_dir, 'bayes_comparison_results.csv'),
        index=False
    )


def run_mlp_comp(round_id, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
                 train_fit_df, artifacts_dir):
    mlp_comp = build_mlp()

    t0 = now()
    mlp_comp.fit(X_train_fit, y_train_fit)
    fit_dt = now() - t0

    val_scores = sanitize_vector(mlp_comp.predict_proba(X_val)[:, 1])

    t0 = now()
    th = find_optimal_threshold(y_val, val_scores)
    th_dt = now() - t0

    t0 = now()
    test_scores = sanitize_vector(mlp_comp.predict_proba(X_test)[:, 1])
    pred_dt = now() - t0

    pred_test_time_s = th_dt + pred_dt
    y_pred = (test_scores > th).astype(int)

    m = evaluate_model(y_test, y_pred, test_scores, {'model': 'MLP_comp'})

    row = {
        'Round': round_id,
        'train_0': int((train_fit_df['label'] == 0).sum()),
        'train_1': int((train_fit_df['label'] == 1).sum()),
        'threshold': th,
        'fit_time_s': fit_dt,
        'pred_test_time_s': pred_test_time_s,
        'AUROC': m['AUROC'],
        'AUPRC': m['AUPRC'],
        'Accuracy': m['Accuracy'],
        'F1': m['F1'],
        'TNR': m['TNR'],
        'TPR': m['TPR'],
        'PPV': m['PPV'],
        'NPV': m['NPV'],
    }
    mlp_comparison_results.append(row)
    pd.DataFrame(mlp_comparison_results).to_csv(
        os.path.join(artifacts_dir, 'mlp_comparison_results.csv'),
        index=False
    )


# -------------------- 結果累積器 --------------------
ct_only_results = []
rf_only_results = []
ct_rf_results = []

last_ct_scaler = None


# -------------------- 主流程 --------------------
if __name__ == "__main__":

    train_df = pd.read_csv(TRAIN_CUR, low_memory=False).drop(columns=["attack_cat"], errors="ignore")
    val_df   = pd.read_csv(VAL_CUR, low_memory=False).drop(columns=["attack_cat"], errors="ignore")
    test_df  = pd.read_csv(TEST_CUR, low_memory=False).drop(columns=["attack_cat"], errors="ignore")

    # 驗證集 neg:pos 比例
    val_labels = val_df['label'].values
    val_pos = int((val_labels == 1).sum())
    val_neg = int((val_labels == 0).sum())
    if val_pos == 0 or val_neg == 0:
        print("[WARN] Validation set 極端不平衡，採用預設 neg:pos = 30:1")
        val_neg_pos_ratio = 30.0
    else:
        val_neg_pos_ratio = val_neg / val_pos
    print(f"[INFO] Validation neg:pos ratio ≈ {val_neg_pos_ratio:.3f} (neg/pos)")

    # train 快照
    RAW_TRAIN_SNAPSHOT = os.path.join(ARTIFACT_DIR, "raw_train_snapshot.csv")
    if not os.path.exists(RAW_TRAIN_SNAPSHOT):
        try:
            shutil.copy2(TRAIN_CUR, RAW_TRAIN_SNAPSHOT)
        except Exception as e:
            print(f"[WARN] 建立原始 train 快照失敗：{e}")

    # ------------------ 初始採樣 ------------------
    df0 = train_df[train_df['label'] == 0]
    df1 = train_df[train_df['label'] == 1]

    total_init = min(1000, len(train_df))
    init_neg, init_pos = compute_neg_pos_for_total(
        total_inc=total_init,
        neg_pos_ratio=val_neg_pos_ratio,
        max_neg=len(df0),
        max_pos=len(df1)
    )

    if init_neg == 0 and len(df0) > 0:
        init_neg = min(len(df0), total_init - 1)
    if init_pos == 0 and len(df1) > 0:
        init_pos = min(len(df1), total_init - init_neg)

    print(f"[INFO] Initial seed total={init_neg + init_pos}, neg={init_neg}, pos={init_pos}")

    seed0 = df0.sample(n=init_neg, random_state=7709)
    seed1 = df1.sample(n=init_pos, random_state=7709)
    train_fit_df = pd.concat([seed0, seed1]).sample(frac=1.0, random_state=7709).copy()

    # 剩餘樣本池
    remainder_pool = train_df.drop(index=train_fit_df.index)

    # X / y
    X_train_fit = sanitize_matrix(train_fit_df.drop(columns=['label']))
    y_train_fit = train_fit_df['label'].values
    X_val = sanitize_matrix(val_df.drop(columns=['label']))
    y_val = val_df['label'].values
    X_test = sanitize_matrix(test_df.drop(columns=['label']))
    y_test = test_df['label'].values

    # 初始化模型
    rf = build_rf()

    # RF 只在初始化 + 每10輪更新
    t0 = now()
    rf.fit(X_train_fit, y_train_fit)
    rf_fit_time_s = now() - t0

    rf_val = sanitize_vector(rf.predict_proba(X_val)[:, 1])
    t0 = now()
    rf_th = find_optimal_threshold(y_val, rf_val)
    rf_th_time = now() - t0

    # Bayes / MLP 一開始也先訓練一次，供第0輪使用
    bayes = build_bayes()
    mlp = build_mlp()

    t0 = now()
    bayes.fit(X_train_fit, y_train_fit)
    bayes_fit_time_s = now() - t0

    t0 = now()
    mlp.fit(X_train_fit, y_train_fit)
    mlp_fit_time_s = now() - t0

    bayes_val = sanitize_vector(bayes.predict_proba(X_val)[:, 1])
    mlp_val = sanitize_vector(mlp.predict_proba(X_val)[:, 1])

    t0 = now()
    bayes_th = find_optimal_threshold(y_val, bayes_val)
    bayes_th_time = now() - t0

    t0 = now()
    mlp_th = find_optimal_threshold(y_val, mlp_val)
    mlp_th_time = now() - t0

    thresholds = {'rf': rf_th, 'bayes': bayes_th, 'mlp': mlp_th}
    with open(os.path.join(ARTIFACT_DIR, 'thresholds_round0.pkl'), 'wb') as f:
        pickle.dump(thresholds, f)

    # ===== 第一次 CT =====
    ensure_clean_dir(FEATURE_COUNT_DIR)

    ct_stat_time_s = f1.statistic(train_fit_df.copy(), out_dir=FEATURE_COUNT_DIR)

    ratio_ct, black_more = compute_ratio_and_flag(
        y_train_fit,
        ct_balance_enabled=CT_BALANCE_ENABLED
    )
    ct_table, _ = f3.load_ratio_table(
        FEATURE_COUNT_DIR,
        ratio_ct,
        black_more,
        return_time=True,
        version=CT_BALANCE_ENABLED
    )

    ct_val_df, map_val_secs = f3.map_features_to_ct(X_val.copy(), ct_table, return_time=True)
    ct_test_df, map_test_secs = f3.map_features_to_ct(X_test.copy(), ct_table, return_time=True)
    ct_map_time_s = map_val_secs + map_test_secs

    ct_val_raw = -ct_val_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values
    ct_test_raw = -ct_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values

    t0 = now()
    ct_th = find_optimal_threshold(y_val, ct_val_raw)
    ct_conf_test, ct_scaler = minmax_confidence_from_val(ct_val_raw, ct_test_raw, ct_th)
    ct_result_time_s = now() - t0
    ct_total_time_s = ct_stat_time_s + ct_map_time_s + ct_result_time_s
    last_ct_scaler = ct_scaler

    # 測試集預測
    t0 = now()
    rf_test = sanitize_vector(rf.predict_proba(X_test)[:, 1])
    rf_pred_test_time_s = (now() - t0) + rf_th_time

    t0 = now()
    bayes_test = sanitize_vector(bayes.predict_proba(X_test)[:, 1])
    bayes_pred_test_time_s = (now() - t0) + bayes_th_time

    t0 = now()
    mlp_test = sanitize_vector(mlp.predict_proba(X_test)[:, 1])
    mlp_pred_test_time_s = (now() - t0) + mlp_th_time

    rf_conf_test, rf_scaler = minmax_confidence_from_val(rf_val, rf_test, thresholds['rf'])
    bayes_conf_test, bayes_scaler = minmax_confidence_from_val(bayes_val, bayes_test, thresholds['bayes'])
    mlp_conf_test, mlp_scaler = minmax_confidence_from_val(mlp_val, mlp_test, thresholds['mlp'])

    with open(os.path.join(ARTIFACT_DIR, "rf_scaler.pkl"), "wb") as f:
        pickle.dump(rf_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "bayes_scaler.pkl"), "wb") as f:
        pickle.dump(bayes_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "mlp_scaler.pkl"), "wb") as f:
        pickle.dump(mlp_scaler, f)

    def log_round(round_id):
        common = {
            'Round': round_id,
            'train_0': int((train_fit_df['label'] == 0).sum()),
            'train_1': int((train_fit_df['label'] == 1).sum()),
            'ct_thres': ct_th,
            'rf_thres': thresholds['rf'],
            'bayes_thres': thresholds['bayes'],
            'mlp_thres': thresholds['mlp'],
            'refresh_rf': (round_id % 10 == 0),
            'refresh_bayes': True,
            'refresh_mlp': True,
            'ct_stat_time_s': ct_stat_time_s,
            'ct_map_time_s': ct_map_time_s,
            'ct_result_time_s': ct_result_time_s,
            'ct_total_time_s': ct_total_time_s,
            'rf_fit_time_s': rf_fit_time_s,
            'rf_pred_test_time_s': rf_pred_test_time_s,
            'bayes_fit_time_s': bayes_fit_time_s,
            'bayes_pred_test_time_s': bayes_pred_test_time_s,
            'mlp_fit_time_s': mlp_fit_time_s,
            'mlp_pred_test_time_s': mlp_pred_test_time_s,
            'ct_balance_enabled': CT_BALANCE_ENABLED,
        }

        # RF only
        rf_bin = (rf_conf_test > 0).astype(int)
        m_rf = evaluate_model(y_test, rf_bin, rf_conf_test, {'model': 'RF'})
        rf_only_results.append({**common, **m_rf})

        # CT only
        ct_bin = (ct_conf_test > 0).astype(int)
        m_ct = evaluate_model(y_test, ct_bin, ct_conf_test, {'ct_selected': len(y_test)})
        ct_only_results.append({**common, **m_ct})

        # CT + RF
        pick_cr = np.where(np.abs(ct_conf_test) >= np.abs(rf_conf_test), ct_conf_test, rf_conf_test)
        bin_cr = (pick_cr > 0).astype(int)
        sel_cr = count_model_selections([ct_conf_test, rf_conf_test], names=['ct', 'rf'])
        m_cr = evaluate_model(y_test, bin_cr, pick_cr, sel_cr)
        ct_rf_results.append({**common, **m_cr})

        # 寫檔
        pd.DataFrame(rf_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'rf_only_results.csv'), index=False)
        pd.DataFrame(ct_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_only_results.csv'), index=False)
        pd.DataFrame(ct_rf_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_rf_results.csv'), index=False)

    # 第 0 輪
    log_round(0)

    # comparison：第 0 輪
    run_bayes_comp(
        0, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
        train_fit_df, ARTIFACT_DIR
    )
    run_mlp_comp(
        0, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
        train_fit_df, ARTIFACT_DIR
    )

    # ===== 增量迭代 =====
    remaining = remainder_pool.copy()
    rem_0 = remaining[remaining['label'] == 0].copy()
    rem_1 = remaining[remaining['label'] == 1].copy()

    round_id = 1
    while len(rem_0) > 0 and len(rem_1) > 0:
        current_size = len(train_fit_df)
        total_inc = get_increment_size(current_size)

        remaining_total = len(rem_0) + len(rem_1)
        total_inc = min(total_inc, remaining_total)
        if total_inc <= 0:
            break

        desired_neg, desired_pos = compute_neg_pos_for_total(
            total_inc=total_inc,
            neg_pos_ratio=val_neg_pos_ratio,
            max_neg=len(rem_0),
            max_pos=len(rem_1)
        )

        if desired_neg == 0 or desired_pos == 0:
            if len(rem_0) == 0 or len(rem_1) == 0:
                break
            desired_pos = min(len(rem_1), max(1, int(round(total_inc / (val_neg_pos_ratio + 1.0)))))
            desired_neg = min(len(rem_0), total_inc - desired_pos)
            if desired_neg <= 0 or desired_pos <= 0:
                break

        batch_pos = rem_1.sample(n=desired_pos, random_state=round_id)
        batch_neg = rem_0.sample(n=desired_neg, random_state=round_id)
        new_batch = pd.concat([batch_neg, batch_pos]).sample(frac=1.0, random_state=round_id)

        print(f"[Round {round_id}] +{len(new_batch)} (neg={desired_neg}, pos={desired_pos}), current total = {current_size + len(new_batch)}")

        train_fit_df = pd.concat([train_fit_df, new_batch], ignore_index=True)
        X_train_fit = sanitize_matrix(train_fit_df.drop(columns=['label']))
        y_train_fit = train_fit_df['label'].values

        # CT recompute
        _, ct_stat_time_s = f2.recompute(train_fit_df, feature_count_dir=FEATURE_COUNT_DIR)

        ratio_ct, black_more = compute_ratio_and_flag(
            y_train_fit,
            ct_balance_enabled=CT_BALANCE_ENABLED
        )
        ct_table, _ = f3.load_ratio_table(
            FEATURE_COUNT_DIR,
            ratio_ct,
            black_more,
            return_time=True,
            version=CT_BALANCE_ENABLED
        )

        ct_val_df, map_val_secs = f3.map_features_to_ct(X_val.copy(), ct_table, return_time=True)
        ct_test_df, map_test_secs = f3.map_features_to_ct(X_test.copy(), ct_table, return_time=True)
        ct_map_time_s = map_val_secs + map_test_secs

        ct_val_raw = -ct_val_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values
        ct_test_raw = -ct_test_df.replace([np.inf, -np.inf], np.nan).fillna(0.0).sum(axis=1).values

        t0 = now()
        ct_th = find_optimal_threshold(y_val, ct_val_raw)
        ct_conf_test, ct_scaler = minmax_confidence_from_val(ct_val_raw, ct_test_raw, ct_th)
        ct_result_time_s = now() - t0
        ct_total_time_s = ct_stat_time_s + ct_map_time_s + ct_result_time_s
        last_ct_scaler = ct_scaler

        # RF 每10輪更新；Bayes/MLP 每輪更新
        rf_fit_time_s = 0.0
        rf_th_time = 0.0

        bayes_fit_time_s = 0.0
        bayes_th_time = 0.0
        mlp_fit_time_s = 0.0
        mlp_th_time = 0.0

        if round_id % 10 == 0:
            t0 = now()
            rf.fit(X_train_fit, y_train_fit)
            rf_fit_time_s = now() - t0

            rf_val = sanitize_vector(rf.predict_proba(X_val)[:, 1])

            t0 = now()
            thresholds['rf'] = find_optimal_threshold(y_val, rf_val)
            rf_th_time = now() - t0

        # Bayes 每輪更新
        bayes = build_bayes()
        t0 = now()
        bayes.fit(X_train_fit, y_train_fit)
        bayes_fit_time_s = now() - t0

        bayes_val = sanitize_vector(bayes.predict_proba(X_val)[:, 1])

        t0 = now()
        thresholds['bayes'] = find_optimal_threshold(y_val, bayes_val)
        bayes_th_time = now() - t0

        # MLP 每輪更新
        mlp = build_mlp()
        t0 = now()
        mlp.fit(X_train_fit, y_train_fit)
        mlp_fit_time_s = now() - t0

        mlp_val = sanitize_vector(mlp.predict_proba(X_val)[:, 1])

        t0 = now()
        thresholds['mlp'] = find_optimal_threshold(y_val, mlp_val)
        mlp_th_time = now() - t0

        with open(os.path.join(ARTIFACT_DIR, f'thresholds_round{round_id}.pkl'), 'wb') as f:
            pickle.dump(thresholds, f)

        # 預測
        # RF 用舊模型或每10輪更新後模型
        rf_val = sanitize_vector(rf.predict_proba(X_val)[:, 1])
        t0 = now()
        rf_test = sanitize_vector(rf.predict_proba(X_test)[:, 1])
        rf_pred_test_time_s = (now() - t0) + rf_th_time
        rf_conf_test, rf_scaler = minmax_confidence_from_val(rf_val, rf_test, thresholds['rf'])

        t0 = now()
        bayes_test = sanitize_vector(bayes.predict_proba(X_test)[:, 1])
        bayes_pred_test_time_s = (now() - t0) + bayes_th_time
        bayes_conf_test, bayes_scaler = minmax_confidence_from_val(bayes_val, bayes_test, thresholds['bayes'])

        t0 = now()
        mlp_test = sanitize_vector(mlp.predict_proba(X_test)[:, 1])
        mlp_pred_test_time_s = (now() - t0) + mlp_th_time
        mlp_conf_test, mlp_scaler = minmax_confidence_from_val(mlp_val, mlp_test, thresholds['mlp'])

        def log_round_k():
            common = {
                'Round': round_id,
                'train_0': int((train_fit_df['label'] == 0).sum()),
                'train_1': int((train_fit_df['label'] == 1).sum()),
                'ct_thres': ct_th,
                'rf_thres': thresholds['rf'],
                'bayes_thres': thresholds['bayes'],
                'mlp_thres': thresholds['mlp'],
                'refresh_rf': (round_id % 10 == 0),
                'refresh_bayes': True,
                'refresh_mlp': True,
                'ct_stat_time_s': ct_stat_time_s,
                'ct_map_time_s': ct_map_time_s,
                'ct_result_time_s': ct_result_time_s,
                'ct_total_time_s': ct_total_time_s,
                'rf_fit_time_s': rf_fit_time_s,
                'rf_pred_test_time_s': rf_pred_test_time_s,
                'bayes_fit_time_s': bayes_fit_time_s,
                'bayes_pred_test_time_s': bayes_pred_test_time_s,
                'mlp_fit_time_s': mlp_fit_time_s,
                'mlp_pred_test_time_s': mlp_pred_test_time_s,
                'ct_balance_enabled': CT_BALANCE_ENABLED,
            }

            # RF only
            rf_bin = (rf_conf_test > 0).astype(int)
            m_rf = evaluate_model(y_test, rf_bin, rf_conf_test, {'model': 'RF'})
            rf_only_results.append({**common, **m_rf})

            # CT only
            ct_bin = (ct_conf_test > 0).astype(int)
            m_ct = evaluate_model(y_test, ct_bin, ct_conf_test, {'ct_selected': len(y_test)})
            ct_only_results.append({**common, **m_ct})

            # CT + RF
            pick_cr = np.where(np.abs(ct_conf_test) >= np.abs(rf_conf_test), ct_conf_test, rf_conf_test)
            bin_cr = (pick_cr > 0).astype(int)
            sel_cr = count_model_selections([ct_conf_test, rf_conf_test], names=['ct', 'rf'])
            m_cr = evaluate_model(y_test, bin_cr, pick_cr, sel_cr)
            ct_rf_results.append({**common, **m_cr})

            pd.DataFrame(rf_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'rf_only_results.csv'), index=False)
            pd.DataFrame(ct_only_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_only_results.csv'), index=False)
            pd.DataFrame(ct_rf_results).to_csv(os.path.join(ARTIFACT_DIR, 'ct_rf_results.csv'), index=False)

        log_round_k()

        # 比較模型：每輪都重訓
        run_bayes_comp(
            round_id, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
            train_fit_df, ARTIFACT_DIR
        )
        run_mlp_comp(
            round_id, X_train_fit, y_train_fit, X_val, y_val, X_test, y_test,
            train_fit_df, ARTIFACT_DIR
        )

        rem_0 = rem_0.drop(batch_neg.index, errors='ignore')
        rem_1 = rem_1.drop(batch_pos.index, errors='ignore')

        round_id += 1

    if last_ct_scaler is not None:
        with open(os.path.join(ARTIFACT_DIR, "ct_scaler.pkl"), "wb") as f:
            pickle.dump(last_ct_scaler, f)

    print("程式執行完成")