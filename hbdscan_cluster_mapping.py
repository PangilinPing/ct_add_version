# -*- coding: utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd
import hdbscan
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ============================================
# CONFIG
# ============================================
TRAIN_CSV = "dataset/NB15_small/train_ori.csv"
LABEL_COL = "label"

FEATURE_COUNT_DIR = "feature_count"
OUTPUT_DIR = "internal_ct_cluster_test"
CLUSTER_STATS_DIR = os.path.join(OUTPUT_DIR, "cluster_stats_per_feature")

# 保留參數（目前不使用 threshold 作強群判斷）
CLUSTER_THRESHOLD = 0.01

# 若某 cluster 在 test 中 unmapped 比例 > 此值，強制不用 mean（保留原始 CT）
MISSING_CLUSTER_THRESHOLD = 0.60

# 強群 mean 門檻：|mean| > threshold 才算 strong
STRONG_MEAN_THRESHOLD = 0.01

# 取樣總數，切一半做 cluster_train / internal_test
SAMPLE_TOTAL = 20000

# unmapped raw value 的 CT 預設值（先用 -1，最後可轉 0）
DEFAULT_CT_VALUE = -1

# 最後把所有 DEFAULT_CT_VALUE(-1) 轉成 0
FINAL_DEFAULT_TO_ZERO = True

# HDBSCAN params（每個 feature 做 1D clustering）
HDB_MIN_CLUSTER_SIZE = 300
HDB_MIN_SAMPLES = 50

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_COUNT_DIR, exist_ok=True)
os.makedirs(CLUSTER_STATS_DIR, exist_ok=True)

# cluster 內部暫存 statistic 的 temp dir（不影響你原輸出）
TMP_CLUSTER_DIR = os.path.join(OUTPUT_DIR, "_tmp_cluster_stat")
os.makedirs(TMP_CLUSTER_DIR, exist_ok=True)

# ============================================
# NEW: missing ratio 五區段設定
# ============================================
MISSING_RATIO_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0000001)]
MISSING_RATIO_BIN_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
MISSING_RATIO_BIN_COLS = [
    "missing_bin_0_20_ratio",
    "missing_bin_20_40_ratio",
    "missing_bin_40_60_ratio",
    "missing_bin_60_80_ratio",
    "missing_bin_80_100_ratio",
]
STACK_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

# ============================================
# Helpers
# ============================================
def sanitize_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    inf/NaN -> 0
    注意：0 會參與 raw clustering
    """
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.fillna(0.0)

def compute_ratio_and_flag(y):
    """
    ratio：黑白比（多數 / 少數）
    black_more：True 表示 label=1 比 label=0 多
    """
    y = np.asarray(y)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 >= n0:
        return n1 / max(n0, 1), True
    else:
        return n0 / max(n1, 1), False

def safe_pie(values, labels, out_path, title):
    """
    避免 NaN/0 導致 matplotlib pie 爆炸
    """
    v = np.array(values, dtype=float)
    if np.any(~np.isfinite(v)) or v.sum() <= 0:
        plt.figure(figsize=(4, 4))
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return

    plt.figure(figsize=(4, 4))
    plt.pie(v, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def series_raw_to_ct(series: pd.Series, ct_map: dict, default_value: float):
    """
    raw -> CT（用 ct_map），unmapped -> default_value
    注意：ct_map 的 key 是字串
    """
    s = series.astype(str)
    ct = s.map(ct_map)
    ct = ct.astype(float)
    ct = ct.fillna(default_value)
    return ct

def series_raw_to_ct_nan(series: pd.Series, ct_map: dict):
    """
    raw -> CT（unmapped -> NaN）
    用在判定 unmapped 缺值
    """
    s = series.astype(str)
    ct = s.map(ct_map)
    return ct.astype(float)  # unmapped -> NaN

def _bin_index(miss_ratio: float) -> int:
    """
    miss_ratio in [0,1]
    """
    for i, (lo, hi) in enumerate(MISSING_RATIO_BINS):
        if (miss_ratio >= lo) and (miss_ratio < hi):
            return i
    return len(MISSING_RATIO_BINS) - 1

# ============================================
# CT mapping imports (你的原模組)
# ============================================
from ct_value.f1_statistic import statistic
from ct_value.f3_mapping import load_ratio_table

# ============================================
# cluster 內重新 statistic + mapping（跳過缺值）再算 mean
# ============================================
def compute_cluster_mean_ct_by_restat(
    original_train_df: pd.DataFrame,
    feature_col: str,
    cluster_train_indices: np.ndarray,
    ratio: float,
    black_more: bool,
    default_ct_value: float,
    tmp_root: str
):
    if len(cluster_train_indices) == 0:
        return None, 0, 0

    cluster_df = original_train_df.iloc[cluster_train_indices].copy()

    temp_dir = os.path.join(
        tmp_root,
        f"{feature_col}__cid_{len(cluster_train_indices)}__{np.random.randint(1e9)}"
    )
    os.makedirs(temp_dir, exist_ok=True)

    try:
        statistic(cluster_df, out_dir=temp_dir)
        cluster_ct_table = load_ratio_table(temp_dir, ratio, black_more)

        ct_map = cluster_ct_table.get(feature_col, {})
        if not isinstance(ct_map, dict) or len(ct_map) == 0:
            return None, 0, int(len(cluster_df))

        ct_series = series_raw_to_ct(cluster_df[feature_col], ct_map, default_ct_value)

        valid = ct_series[(ct_series != default_ct_value) & np.isfinite(ct_series)]
        valid_count = int(valid.shape[0])
        total_count = int(ct_series.shape[0])
        if valid_count == 0:
            return None, 0, total_count

        mean_ct = float(valid.mean())
        return mean_ct, valid_count, total_count

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ============================================
# 核心：raw -> cluster -> cluster_mean_CT -> strong? -> replace
# + 追加：per-feature missing 所在 cluster miss_ratio 五區段比例
# ============================================
def fused_ct_by_raw_hdbscan(
    original_train_df: pd.DataFrame,
    X_train_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    ct_table: dict,
    default_ct_value: float,
    threshold: float,  # 保留
    missing_cluster_threshold: float,
    hdb_min_cluster_size: int,
    hdb_min_samples: int,
    cluster_stats_dir: str,
    ratio: float,
    black_more: bool,
    strong_mean_threshold: float,
    tmp_root: str
):
    ct_original_test = pd.DataFrame(index=X_test_raw.index)
    ct_fused_test = pd.DataFrame(index=X_test_raw.index)

    feature_rows = []

    total_missing = 0
    total_missing_converted = 0
    total_cells = int(X_test_raw.shape[0] * X_test_raw.shape[1])

    total_missing_forced_due_to_30 = 0
    total_missing_noise = 0

    total_strong_cells = 0
    total_forced_split_cells = 0

    for col in X_train_raw.columns:
        if col not in X_test_raw.columns:
            continue

        tr_raw = X_train_raw[col]
        te_raw = X_test_raw[col]

        # ============ RAW clustering ============
        tr_x = pd.to_numeric(tr_raw, errors="coerce").fillna(0.0).values.reshape(-1, 1)
        te_x = pd.to_numeric(te_raw, errors="coerce").fillna(0.0).values.reshape(-1, 1)

        hdb = hdbscan.HDBSCAN(
            min_cluster_size=hdb_min_cluster_size,
            min_samples=hdb_min_samples,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True
        )
        hdb.fit(tr_x)

        tr_labels, _ = hdbscan.approximate_predict(hdb, tr_x)
        te_labels, _ = hdbscan.approximate_predict(hdb, te_x)

        # ============ 原始 CT（test）===========
        global_ct_map = ct_table.get(col, {})
        te_ct_original = series_raw_to_ct(te_raw, global_ct_map, default_ct_value)
        ct_original_test[col] = te_ct_original.values

        # unmapped 缺值（用 global mapping 判斷）
        te_ct_nan = series_raw_to_ct_nan(te_raw, global_ct_map)  # unmapped -> NaN
        missing_mask = te_ct_nan.isna().values
        missing_cnt = int(missing_mask.sum())
        total_missing += missing_cnt

        # 缺值中 noise(-1 cluster) 統計
        noise_mask = (te_labels == -1)
        missing_noise_cnt = int(np.sum(missing_mask & noise_mask))
        total_missing_noise += missing_noise_cnt

        # ============ test cluster size/missing ============
        df_te_stat = pd.DataFrame({
            "cluster": te_labels,
            "is_missing_unmapped": missing_mask.astype(int),
            "ct_original": te_ct_original.values
        })
        cluster_size_test = df_te_stat.groupby("cluster").size().to_dict()
        cluster_missing_test = df_te_stat.groupby("cluster")["is_missing_unmapped"].sum().to_dict()

        # ============ train cluster indices mapping ============
        df_tr_label = pd.DataFrame({"cluster": tr_labels})
        cluster_to_train_iloc = {}
        for cid, sub in df_tr_label.groupby("cluster"):
            cluster_to_train_iloc[int(cid)] = sub.index.values  # positional indices for iloc

        # ============ 每個 cluster 的 mean（群內重算 statistic+mapping）===========
        cluster_mean_train = {}
        cluster_train_valid_ct_count = {}
        cluster_train_size = {}

        for cid, iloc_idx in cluster_to_train_iloc.items():
            cluster_train_size[cid] = int(len(iloc_idx))

            mean_ct, valid_ct_cnt, _ = compute_cluster_mean_ct_by_restat(
                original_train_df=original_train_df,
                feature_col=col,
                cluster_train_indices=iloc_idx,
                ratio=ratio,
                black_more=black_more,
                default_ct_value=default_ct_value,
                tmp_root=tmp_root
            )

            cluster_train_valid_ct_count[cid] = int(valid_ct_cnt)
            if mean_ct is None or (not np.isfinite(mean_ct)):
                continue
            cluster_mean_train[cid] = float(mean_ct)

        # ============ strong cluster ============
        strong_clusters = {
            cid for cid, m in cluster_mean_train.items()
            if abs(m) > strong_mean_threshold
        }

        # ============ 融合 CT（test）===========
        fused = te_ct_original.copy()

        converted_cnt = 0
        strong_cells = 0
        forced_split_cells = 0
        missing_forced_30_cnt = 0

        for cid in strong_clusters:
            mean_ct = cluster_mean_train.get(cid, None)
            if mean_ct is None:
                continue

            mask_cluster = (te_labels == cid)
            if not np.any(mask_cluster):
                continue

            size_test = int(cluster_size_test.get(cid, int(np.sum(mask_cluster))))
            miss_test = int(cluster_missing_test.get(cid, int(np.sum(mask_cluster & missing_mask))))
            miss_ratio = (miss_test / size_test) if size_test > 0 else 0.0

            if miss_ratio > missing_cluster_threshold:
                forced_split_cells += size_test
                missing_forced_30_cnt += int(np.sum(mask_cluster & missing_mask))
                continue

            converted_cnt += int(np.sum(mask_cluster & missing_mask))
            fused.loc[mask_cluster] = mean_ct
            strong_cells += int(np.sum(mask_cluster))

        total_missing_converted += converted_cnt
        total_strong_cells += strong_cells
        total_forced_split_cells += forced_split_cells
        total_missing_forced_due_to_30 += missing_forced_30_cnt

        ct_fused_test[col] = fused.values

        # ============ per-feature cluster stats CSV ============
        df_te_stat["ct_fused"] = fused.values

        rows_cluster = []
        for cid in sorted(cluster_size_test.keys(), key=lambda x: int(x)):
            cid_int = int(cid)
            size = int(cluster_size_test[cid])
            miss = int(cluster_missing_test.get(cid, 0))
            miss_ratio = (miss / size) if size > 0 else 0.0

            mean_ct_train = cluster_mean_train.get(cid_int, np.nan)
            is_strong = int(cid_int in strong_clusters)
            force_use_original = int(is_strong == 1 and miss_ratio > missing_cluster_threshold)

            sub = df_te_stat[df_te_stat["cluster"] == cid_int]
            ct_mean_test_original = float(sub["ct_original"].mean()) if len(sub) > 0 else np.nan
            ct_mean_test_fused = float(sub["ct_fused"].mean()) if len(sub) > 0 else np.nan

            rows_cluster.append({
                "feature": col,
                "cluster_id": cid_int,
                "test_cluster_size": size,
                "test_missing_unmapped": miss,
                "test_missing_ratio": miss_ratio,
                "train_cluster_size": int(cluster_train_size.get(cid_int, 0)),
                "train_valid_ct_count": int(cluster_train_valid_ct_count.get(cid_int, 0)),
                "cluster_ct_mean_train": mean_ct_train,
                "cluster_ct_mean_test_original": ct_mean_test_original,
                "cluster_ct_mean_test_fused": ct_mean_test_fused,
                "is_strong": is_strong,
                "force_use_original_due_to_missing": force_use_original,
            })

        out_cluster_path = os.path.join(cluster_stats_dir, f"{col}.csv")
        pd.DataFrame(rows_cluster).to_csv(out_cluster_path, index=False)

        # ============================================
        # NEW: per-feature「missing 所在 cluster missing_ratio」五區段比例
        # ============================================
        bin_missing_counts = np.zeros(5, dtype=float)

        if missing_cnt > 0:
            # 只針對 missing 的樣本，看它所在 cluster 的 miss_ratio 屬於哪個區段
            # 先算出每個 cluster 的 miss_ratio
            cluster_miss_ratio = {}
            for cid_k, size_k in cluster_size_test.items():
                size_k = int(size_k)
                miss_k = int(cluster_missing_test.get(cid_k, 0))
                r_k = (miss_k / size_k) if size_k > 0 else 0.0
                cluster_miss_ratio[int(cid_k)] = float(r_k)

            missing_clusters = te_labels[missing_mask]
            # 對 missing 的每筆資料，找到它所在 cluster 的 ratio -> bin
            for cid_k in missing_clusters:
                cid_k = int(cid_k)
                r_k = cluster_miss_ratio.get(cid_k, 0.0)
                bidx = _bin_index(r_k)
                bin_missing_counts[bidx] += 1.0

        bin_missing_ratios = (bin_missing_counts / missing_cnt) if missing_cnt > 0 else np.zeros(5, dtype=float)

        # ============ feature 統計 ============
        converted_ratio = (converted_cnt / missing_cnt) if missing_cnt > 0 else 0.0
        missing_ratio_feature = missing_cnt / max(len(te_raw), 1)
        risk = missing_ratio_feature * (1.0 - converted_ratio)

        row = {
            "feature": col,
            "test_rows": int(len(te_raw)),

            "missing_unmapped_total": missing_cnt,
            "missing_unmapped_ratio": float(missing_ratio_feature),

            "missing_converted_to_ctmean": int(converted_cnt),
            "missing_converted_ratio": float(converted_ratio),

            "strong_cluster_count": int(len(strong_clusters)),
            "clusters_with_mean": int(len(cluster_mean_train)),

            "strong_cells_covered": int(strong_cells),
            "strong_cells_ratio": float(strong_cells / max(len(te_raw), 1)),

            "risk": float(risk),

            "forced_split_cells": int(forced_split_cells),
            "forced_split_ratio": float(forced_split_cells / max(len(te_raw), 1)),

            "missing_noise_count": int(missing_noise_cnt),
            "missing_noise_ratio": float(missing_noise_cnt / missing_cnt) if missing_cnt > 0 else 0.0,

            "missing_forced_due_to_30_count": int(missing_forced_30_cnt),
            "missing_forced_due_to_30_ratio": float(missing_forced_30_cnt / missing_cnt) if missing_cnt > 0 else 0.0,
        }

        # 五區段比例欄位
        for i, c in enumerate(MISSING_RATIO_BIN_COLS):
            row[c] = float(bin_missing_ratios[i])

        feature_rows.append(row)

    feature_stats_df = pd.DataFrame(feature_rows)
    if len(feature_stats_df) > 0:
        feature_stats_df = feature_stats_df.sort_values(
            ["risk", "missing_unmapped_total"],
            ascending=False
        )

    total_missing_remain = int(total_missing - total_missing_converted - total_missing_forced_due_to_30)
    if total_missing_remain < 0:
        total_missing_remain = 0

    global_stats = {
        "internal_test_rows": int(X_test_raw.shape[0]),
        "internal_test_cols": int(X_test_raw.shape[1]),
        "total_cells": int(total_cells),

        "total_missing_unmapped_before": int(total_missing),
        "total_missing_converted_to_ctmean": int(total_missing_converted),
        "missing_converted_ratio": (total_missing_converted / total_missing) if total_missing > 0 else 0.0,

        "total_strong_cells_covered": int(total_strong_cells),
        "strong_cells_ratio": (total_strong_cells / total_cells) if total_cells > 0 else 0.0,

        "total_forced_split_cells": int(total_forced_split_cells),
        "forced_split_ratio": (total_forced_split_cells / total_cells) if total_cells > 0 else 0.0,

        "missing_cluster_threshold": float(missing_cluster_threshold),
        "strong_mean_threshold": float(strong_mean_threshold),

        "total_missing_forced_due_to_30": int(total_missing_forced_due_to_30),
        "missing_forced_due_to_30_ratio": (total_missing_forced_due_to_30 / total_missing) if total_missing > 0 else 0.0,

        "total_missing_noise": int(total_missing_noise),
        "missing_noise_ratio": (total_missing_noise / total_missing) if total_missing > 0 else 0.0,

        "total_missing_remain": int(total_missing_remain),
        "missing_remain_ratio": (total_missing_remain / total_missing) if total_missing > 0 else 0.0,
    }

    pie_stats = {
        "missing_converted": int(total_missing_converted),
        "missing_forced_30": int(total_missing_forced_due_to_30),
        "missing_remain": int(total_missing_remain),
        "missing_total": int(total_missing),
    }

    noise_stats = {
        "missing_noise": int(total_missing_noise),
        "missing_non_noise": int(total_missing - total_missing_noise),
        "missing_total": int(total_missing)
    }

    return ct_original_test, ct_fused_test, feature_stats_df, global_stats, pie_stats, noise_stats


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV)

    # 抽樣後切半
    df_sample = df.sample(n=SAMPLE_TOTAL, random_state=RANDOM_STATE).reset_index(drop=True)

    df_cluster_train, df_internal_test = train_test_split(
        df_sample,
        test_size=0.5,
        random_state=RANDOM_STATE
    )
    df_cluster_train = df_cluster_train.reset_index(drop=True)
    df_internal_test = df_internal_test.reset_index(drop=True)

    # raw X
    X_cluster_raw = sanitize_matrix(df_cluster_train.drop(columns=[LABEL_COL]))
    X_internal_raw = sanitize_matrix(df_internal_test.drop(columns=[LABEL_COL]))

    y_cluster = df_cluster_train[LABEL_COL].values
    ratio, black_more = compute_ratio_and_flag(y_cluster)

    # CT statistic + table（只從 cluster_train 建表）
    print("Running CT statistic on cluster_train...")
    statistic(df_cluster_train.copy(), out_dir=FEATURE_COUNT_DIR)

    ct_table = load_ratio_table(FEATURE_COUNT_DIR, ratio, black_more)

    # Raw clustering + fused CT
    print("Raw clustering + fused CT mapping...")
    ct_original, ct_fused, feature_stats_df, global_stats, pie_stats, noise_stats = fused_ct_by_raw_hdbscan(
        original_train_df=df_cluster_train,
        X_train_raw=X_cluster_raw,
        X_test_raw=X_internal_raw,
        ct_table=ct_table,
        default_ct_value=DEFAULT_CT_VALUE,
        threshold=CLUSTER_THRESHOLD,
        missing_cluster_threshold=MISSING_CLUSTER_THRESHOLD,
        hdb_min_cluster_size=HDB_MIN_CLUSTER_SIZE,
        hdb_min_samples=HDB_MIN_SAMPLES,
        cluster_stats_dir=CLUSTER_STATS_DIR,
        ratio=ratio,
        black_more=black_more,
        strong_mean_threshold=STRONG_MEAN_THRESHOLD,
        tmp_root=TMP_CLUSTER_DIR
    )

    # 最後把 -1 -> 0
    ct_fused_final = ct_fused.copy()
    if FINAL_DEFAULT_TO_ZERO:
        ct_fused_final = ct_fused_final.replace(DEFAULT_CT_VALUE, 0.0)

    # Save CSV（檔名不改）
    out_original = os.path.join(OUTPUT_DIR, "ct_internal_original.csv")
    out_transformed = os.path.join(OUTPUT_DIR, "ct_internal_transformed.csv")
    out_feature_stats = os.path.join(OUTPUT_DIR, "feature_missing_conversion_stats.csv")
    out_global = os.path.join(OUTPUT_DIR, "internal_global_summary.csv")

    ct_original.to_csv(out_original, index=False)
    ct_fused_final.to_csv(out_transformed, index=False)
    feature_stats_df.to_csv(out_feature_stats, index=False)
    pd.DataFrame([global_stats]).to_csv(out_global, index=False)

    print("Saved:")
    print(" -", out_original)
    print(" -", out_transformed)
    print(" -", out_feature_stats)
    print(" -", out_global)
    print(" - per-feature cluster stats ->", CLUSTER_STATS_DIR)

    # ============================================
    # ============================================
    # Visualization（最終正確版本）
    # ============================================

    total_missing = global_stats["total_missing_unmapped_before"]
    converted = global_stats["total_missing_converted_to_ctmean"]
    remain = total_missing - converted

    # 1) 2-way count
    plt.figure(figsize=(6, 4))
    plt.bar(["Converted", "Remain"], [converted, remain])
    plt.ylabel("Count (unmapped raw values)")
    plt.title("Unmapped Raw Values: Converted by Strong Clusters")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_converted_bar.png"))
    plt.close()

    safe_pie(
        [converted, remain],
        ["Converted", "Remain"],
        os.path.join(OUTPUT_DIR, "missing_converted_pie.png"),
        "Unmapped Conversion Ratio"
    )

    if len(feature_stats_df) > 0:

        df_plot = feature_stats_df.copy()
        df_plot = df_plot.reset_index(drop=True)

        # --------------------------------------------------
        # 2) per-feature 轉換比例（保留）
        # --------------------------------------------------
        plt.figure(figsize=(12, 6))
        plt.bar(df_plot["feature"], df_plot["missing_converted_ratio"])
        plt.xticks(rotation=90)
        plt.ylabel("Converted Ratio")
        plt.title("Per-Feature Unmapped Conversion Ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "per_feature_missing_converted_ratio.png"))
        plt.close()

        # ==================================================
        # ✅ 每個 feature 一根堆疊柱（數量版）
        # ==================================================
        plt.figure(figsize=(14, 6))

        x = np.arange(len(df_plot))
        bottoms = np.zeros(len(df_plot))

        for i, col_bin in enumerate(MISSING_RATIO_BIN_COLS):
            values = (
                df_plot[col_bin].values *
                df_plot["missing_unmapped_total"].values
            )
            plt.bar(
                x,
                values,
                bottom=bottoms,
                color=STACK_COLORS[i],
                label=MISSING_RATIO_BIN_LABELS[i]
            )
            bottoms += values

        plt.xticks(x, df_plot["feature"], rotation=90)
        plt.ylabel("Missing Count")
        plt.title("Per-Feature Missing Count by Cluster Missing-Ratio (Stacked 5 bins)")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "per_feature_missing_count.png"))
        plt.close()

        # ==================================================
        # ✅ 每個 feature 一根堆疊柱（比例版）
        # ==================================================
        plt.figure(figsize=(14, 6))

        bottoms = np.zeros(len(df_plot))

        for i, col_bin in enumerate(MISSING_RATIO_BIN_COLS):
            values = df_plot[col_bin].values
            plt.bar(
                x,
                values,
                bottom=bottoms,
                color=STACK_COLORS[i],
                label=MISSING_RATIO_BIN_LABELS[i]
            )
            bottoms += values

        plt.xticks(x, df_plot["feature"], rotation=90)
        plt.ylabel("Proportion within Feature Missing")
        plt.title("Per-Feature Missing Ratio by Cluster Missing-Ratio (Stacked 5 bins)")
        plt.ylim(0, 1)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "per_feature_missing_ratio.png"))
        plt.close()

        # --------------------------------------------------
        # Risk ranking（保留）
        # --------------------------------------------------
        plt.figure(figsize=(12, 6))
        plt.bar(df_plot["feature"], df_plot["risk"])
        plt.xticks(rotation=90)
        plt.ylabel("Risk")
        plt.title("Feature Risk Ranking")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "feature_risk_ranking.png"))
        plt.close()

        # --------------------------------------------------
        # Forced split ratio（保留）
        # --------------------------------------------------
        plt.figure(figsize=(12, 6))
        plt.bar(df_plot["feature"], df_plot["forced_split_ratio"])
        plt.xticks(rotation=90)
        plt.ylabel("Forced Split Ratio")
        plt.title("Per-Feature Forced Split Ratio")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "per_feature_forced_split_ratio.png"))
        plt.close()

    # ==================================================
    # 3-way missing 分類
    # ==================================================
    miss_conv = pie_stats["missing_converted"]
    miss_forced30 = pie_stats["missing_forced_30"]
    miss_remain = pie_stats["missing_remain"]

    safe_pie(
        [miss_conv, miss_forced30, miss_remain],
        ["Converted", "Forced (>threshold)", "Remain"],
        os.path.join(OUTPUT_DIR, "missing_converted_pie_3way.png"),
        "Unmapped Handling (3-way)"
    )

    plt.figure(figsize=(6, 4))
    plt.bar(
        ["Converted", "Forced (>threshold)", "Remain"],
        [miss_conv, miss_forced30, miss_remain]
    )
    plt.ylabel("Count (unmapped raw values)")
    plt.title("Unmapped Handling (3-way)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_converted_bar_3way.png"))
    plt.close()

    # ==================================================
    # Noise 統計
    # ==================================================
    noise_cnt = noise_stats["missing_noise"]
    non_noise_cnt = noise_stats["missing_non_noise"]

    plt.figure(figsize=(6, 4))
    plt.bar(["Noise (-1)", "Non-noise"], [noise_cnt, non_noise_cnt])
    plt.ylabel("Count (unmapped raw values)")
    plt.title("Unmapped Values: Noise vs Non-noise")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "missing_noise_bar.png"))
    plt.close()

    safe_pie(
        [noise_cnt, non_noise_cnt],
        ["Noise (-1)", "Non-noise"],
        os.path.join(OUTPUT_DIR, "missing_noise_pie.png"),
        "Unmapped Noise Ratio"
    )

    print("\nDone.")
