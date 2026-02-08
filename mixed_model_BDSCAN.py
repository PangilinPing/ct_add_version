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
import hdbscan



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, roc_curve, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# === 依照你提供的檔案函式介面 ===
import ct_value.f1_statistic as f1   # statistic(df, out_dir='...') -> compute_secs（純統計時間）
import ct_value.f2_recompute as f2   # recompute(df, feature_count_dir='...') -> (tables, compute_secs)；會寫檔
import ct_value.f3_mapping as f3     # load_ratio_table(..., return_time=bool), map_features_to_ct(..., return_time=bool)

# -------------------- 路徑設定 --------------------
ARTIFACT_DIR = "artifacts_NB15_CT_RF_BDSCAN"
FEATURE_COUNT_DIR = os.path.normpath(os.path.join(os.getcwd(), "feature_count"))
DATASET_CUR = os.path.join("dataset", "NB15_small")
TRAIN_CUR = os.path.join(DATASET_CUR, "train.csv")
VAL_CUR   = os.path.join(DATASET_CUR, "val.csv")   # ★ validation 直接讀檔
TEST_CUR  = os.path.join(DATASET_CUR, "test.csv")

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

def evaluate_model(y_true, y_pred, y_score, model_usage=None,):
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
    print(model_usage)
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
    n_black = int((y == 1).sum())
    n_white = int((y == 0).sum())
    if n_black >= n_white:
        black_more = True
        ratio = n_black / max(n_white, 1)
    else:
        black_more = False
        ratio = n_white / max(n_black, 1)
    return ratio, black_more

def Get_Scale_Pos_Weight(sensitivity):
    return sensitivity * 30

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

def train_hdbscan_per_column(
    ct_train_df: pd.DataFrame,
    save_dir: str | None = None,
    min_cluster_size: int = 30,
    min_samples: int = 10
):
    """
    對每個 CT column 各自訓練一個 HDBSCAN（1D CT space）
    
    回傳：
        models: dict[col_name] = hdbscan_model

    若 save_dir != None，會同時把模型存成 pickle
    """

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    models = {}

    for col in ct_train_df.columns:
        print(f"[HDBSCAN] training column: {col}")

        # --- 2. HDBSCAN（不做 scaler） ---
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',              # 1D CT
            cluster_selection_method='eom',
            prediction_data=True
        )

        hdb.fit(ct_train_df[col].values.reshape(-1,1))

        # --- 3. 放進 dict ---
        models[col] = hdb

        # --- 4. （可選）存檔 ---
        if save_dir is not None:
            with open(os.path.join(save_dir, f"{col}.pkl"), "wb") as f:
                pickle.dump(hdb, f)

    # （可選）存一份 index
    if save_dir is not None:
        with open(os.path.join(save_dir, "model_index.pkl"), "wb") as f:
            pickle.dump(list(models.keys()), f)

    return models


# -------------------- 結果累積器 --------------------
ct_only_results = []
ct_xg_results = []
ct_xg_rf_results = []
ct_xg_rf_ada_results = []
xg_only_results = []
xg_rf_results = []
xg_rf_ada_results = []

last_ct_scaler = None

# -------------------- 主流程 --------------------
if __name__ == "__main__":
    # 讀資料
    train_df = pd.read_csv(TRAIN_CUR, low_memory=False)
    val_df   = pd.read_csv(VAL_CUR,   low_memory=False)
    test_df  = pd.read_csv(TEST_CUR,  low_memory=False)

    # train 快照（一次）
    RAW_TRAIN_SNAPSHOT = os.path.join(ARTIFACT_DIR, "raw_train_snapshot.csv")
    if not os.path.exists(RAW_TRAIN_SNAPSHOT):
        try:
            shutil.copy2(TRAIN_CUR, RAW_TRAIN_SNAPSHOT)
        except Exception as e:
            print(f"[WARN] 建立原始 train 快照失敗：{e}")

    # 初始採樣（維持約 30:1）
    df0 = train_df[train_df['label'] == 0]
    df1 = train_df[train_df['label'] == 1]
    seed0 = df0.sample(n=len(df0), random_state=7709)
    seed1 = df1.sample(n=len(df1), random_state=7709)
    train_fit_df = pd.concat([seed0, seed1]).sample(frac=1.0, random_state=7709).copy()

    # 剩餘樣本池（供後續增量）
    remainder_pool = train_df.drop(index=train_fit_df.index)

    # X / y
    X_train_fit = sanitize_matrix(train_fit_df.drop(columns=['label']))
    y_train_fit = train_fit_df['label'].values
    X_val = sanitize_matrix(val_df.drop(columns=['label']))
    y_val = val_df['label'].values
    X_test = sanitize_matrix(test_df.drop(columns=['label']))
    y_test = test_df['label'].values

    # 初始化模型與訓練（fit 計時）
    rf  = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, max_depth=10, n_jobs=None)
    xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100,
                        scale_pos_weight=Get_Scale_Pos_Weight(sensitivity=0.85),
                        use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=None)
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)

    t0=now(); rf.fit(X_train_fit, y_train_fit);   rf_fit_time_s  = now()-t0
    t0=now(); xgb.fit(X_train_fit, y_train_fit);  xg_fit_time_s  = now()-t0
    t0=now(); ada.fit(X_train_fit, y_train_fit);  ada_fit_time_s = now()-t0

    # -------------------- 【新增】存模型（deployment/sniff 用） --------------------
    # 你要 sniff 能直接 load，就一定要把 model 存下來
    with open(os.path.join(ARTIFACT_DIR, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(ARTIFACT_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb, f)
    with open(os.path.join(ARTIFACT_DIR, "ada_model.pkl"), "wb") as f:
        pickle.dump(ada, f)

    # ---- 先在 val 上求各自門檻 ----
    rf_val  = sanitize_vector(rf.predict_proba(X_val)[:, 1])
    xg_val  = sanitize_vector(xgb.predict_proba(X_val)[:, 1])
    ada_val = sanitize_vector(ada.predict_proba(X_val)[:, 1])

    t0=now(); rf_th  = find_optimal_threshold(y_val, rf_val);   rf_th_time  = now()-t0
    t0=now(); xg_th  = find_optimal_threshold(y_val, xg_val);   xg_th_time  = now()-t0
    t0=now(); ada_th = find_optimal_threshold(y_val, ada_val);  ada_th_time = now()-t0

    # 先暫存（之後 CT 算完會一起覆蓋寫入 thresholds_round0.pkl）
    thresholds = {'rf': rf_th, 'xg': xg_th, 'ada': ada_th}

    # ===== 第一次 CT：statistic + mapping + result =====
    ensure_clean_dir(FEATURE_COUNT_DIR)

    # f1：純統計時間
    ct_stat_time_s = f1.statistic(train_fit_df.copy(), out_dir=FEATURE_COUNT_DIR)

    # f3：載入 ratio table
    ratio, black_more = compute_ratio_and_flag(y_train_fit)
    ct_table, _ = f3.load_ratio_table(FEATURE_COUNT_DIR, ratio, black_more, return_time=True)

    ### hdbscan 訓練

    ct_train_df, map_train_secs = f3.map_features_to_ct(
        X_train_fit.copy(),
        ct_table,
        return_time=True
    )

    hdb_models = train_hdbscan_per_column(
        ct_train_df=ct_train_df,
        save_dir=os.path.join(ARTIFACT_DIR, "ct_hdbscan_columns"),
        min_cluster_size=30,
        min_samples=10
    )

    # f3：映射（val + test）
    ct_val_df,  map_val_secs  = f3.map_ct_with_hdbscan_models(X_val.copy(),  ct_table, hdb_models,return_time=True)
    ct_test_df, map_test_secs = f3.map_ct_with_hdbscan_models(X_test.copy(), ct_table, hdb_models,return_time=True)
    ct_map_time_s = map_val_secs + map_test_secs

    # raw 分數
    ct_rf = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42, max_depth=10, n_jobs=None)
    t0=now(); ct_rf.fit(ct_train_df, y_train_fit);  ct_rf_fit_time_s = now()-t0

    # -------------------- 【新增】存模型（deployment/sniff 用） --------------------
    # 你要 sniff 能直接 load，就一定要把 model 存下來
    with open(os.path.join(ARTIFACT_DIR, "ct_rf_model.pkl"), "wb") as f:
        pickle.dump(rf, f)

    # ---- 先在 val 上求各自門檻 ----
    ct_rf_val  = sanitize_vector(ct_rf.predict_proba(ct_val_df)[:, 1])
    ct_rf_test  = sanitize_vector(ct_rf.predict_proba(ct_test_df)[:, 1]); 

    # 結果計算（找 threshold + MinMax→test）
    t0 = now()
    ct_th = find_optimal_threshold(y_val, ct_rf_val)
    ct_conf_test, ct_scaler = minmax_confidence_from_val(ct_rf_val, ct_rf_test, ct_th)
    ct_result_time_s = now() - t0
    ct_total_time_s  = ct_stat_time_s + ct_map_time_s + ct_result_time_s
    last_ct_scaler = ct_scaler

    # -------------------- 【新增】CT threshold 也要存進 thresholds --------------------
    thresholds["ct"] = ct_th

    # -------------------- 【新增】存 thresholds_round0.pkl（含 CT/RF/XGB/ADA） --------------------
    with open(os.path.join(ARTIFACT_DIR, 'thresholds_round0.pkl'), 'wb') as f:
        pickle.dump(thresholds, f)

    # 模型：test 預測（+ 門檻時間）
    t0=now(); rf_test  = sanitize_vector(rf.predict_proba(X_test)[:, 1]);  rf_pred_test_time_s  = (now()-t0) + rf_th_time
    t0=now(); xg_test  = sanitize_vector(xgb.predict_proba(X_test)[:, 1]); xg_pred_test_time_s = (now()-t0) + xg_th_time
    t0=now(); ada_test = sanitize_vector(ada.predict_proba(X_test)[:, 1]);  ada_pred_test_time_s = (now()-t0) + ada_th_time

    # 轉信心值（val→test MinMax）
    rf_conf_test,  rf_scaler  = minmax_confidence_from_val(rf_val,  rf_test,  thresholds['rf'])
    xg_conf_test,  xg_scaler  = minmax_confidence_from_val(xg_val,  xg_test,  thresholds['xg'])
    ada_conf_test, ada_scaler = minmax_confidence_from_val(ada_val, ada_test, thresholds['ada'])

    # -------------------- 【新增】存 scaler（CT/RF/XGB/ADA 全部存） --------------------
    with open(os.path.join(ARTIFACT_DIR, "ct_scaler.pkl"), "wb") as f:
        pickle.dump(ct_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "rf_scaler.pkl"), "wb") as f:
        pickle.dump(rf_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "xg_scaler.pkl"), "wb") as f:
        pickle.dump(xg_scaler, f)
    with open(os.path.join(ARTIFACT_DIR, "ada_scaler.pkl"), "wb") as f:
        pickle.dump(ada_scaler, f)

    # ---- 第 0 輪寫入 ----
    def log_round(round_id):
        common = {
            'Round': round_id,
            'train_0': int((train_fit_df['label'] == 0).sum()),
            'train_1': int((train_fit_df['label'] == 1).sum()),
            'ct_thres': thresholds['ct'],
            'rf_thres': thresholds['rf'],
            'xg_thres': thresholds['xg'],
            'ada_thres': thresholds['ada'],
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
        sel_full = count_model_selections([ct_conf_test, xg_conf_test, rf_conf_test, ada_conf_test],
                                          names=['ct','xg','rf','ada'])
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

    print("程式執行完成")
