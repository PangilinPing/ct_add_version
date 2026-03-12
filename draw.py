import os
import pandas as pd
import matplotlib.pyplot as plt

# 來源與輸出目錄
base_dir   = 'ts_version/artifacts/data'
output_dir = os.path.join(base_dir, 'images')
os.makedirs(output_dir, exist_ok=True)

# ================= AUROC/BA 的資料來源 =================
candidates = [
    ('ct_only_results.csv',        'ST Sum only',                    '#d62728'),
    ('ct_xg_results.csv',          'ST Sum + USXGB',                 '#1f77b4'),
    ('ct_xg_rf_results.csv',       'ST Sum + USXGB + RF',            '#2ca02c'),
    ('ct_xg_rf_ada_results.csv',   'ST Sum + USXGB + RF + ADABoost', '#ff7f0e'),
    ('xgb_comparison_result.csv',  'USXGB baseline',                 '#9467bd'),
    ('xgb_comparison_results.csv', 'USXGB baseline',                 '#9467bd'),
]

# ================= 線條樣式設定 =================
linestyles = {
    'ST Sum only': 'solid',
    'ST Sum + USXGB': 'dashed',
    'ST Sum + USXGB + RF': 'dashdot',
    'ST Sum + USXGB + RF + ADABoost': 'dotted',
    'USXGB baseline': (0, (6, 1, 1, 1, 1, 1, 1, 1, 1, 1)),
}

def _pick_total_train_cols(df: pd.DataFrame) -> pd.Series:
    if {'train_0','train_1'}.issubset(df.columns):
        return df['train_0'] + df['train_1']
    if {'train_0_count','train_1_count'}.issubset(df.columns):
        return df['train_0_count'] + df['train_1_count']
    raise KeyError("缺少 train_0/train_1 欄位（或 *_count 欄位），無法計算訓練樣本總數。")

def _val(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key in row and pd.notna(row[key]):
        try:
            return float(row[key])
        except Exception:
            return default
    return default

# =============== 基礎 AUROC / BalancedAcc ===============
series = []

for fname, label, color in candidates:
    fpath = os.path.join(base_dir, fname)
    if not os.path.exists(fpath):
        continue
    df = pd.read_csv(fpath)
    need_cols = {'AUROC', 'TNR', 'TPR'}
    if not need_cols.issubset(df.columns):
        continue
    try:
        total_train = _pick_total_train_cols(df)
    except KeyError:
        continue
    if 'Round' not in df.columns:
        df['Round'] = range(len(df))
    df['TotalTrain'] = total_train
    df['BalancedAcc'] = (df['TNR'] + df['TPR']) / 2.0
    df = df.sort_values('TotalTrain')
    series.append((label, color, df[['Round','TotalTrain','AUROC','BalancedAcc']]))

# =============== 時間序列設定 ===============
TIME_CONFIGS = [
    ('ST Sum only',                    'ct_only_results.csv',      '#d62728',
     lambda row: _val(row, 'ct_stat_time_s')),
    ('ST Sum + USXGB',                 'ct_xg_results.csv',        '#1f77b4',
     lambda row: _val(row, 'ct_stat_time_s') + _val(row, 'xg_fit_time_s')),
    ('ST Sum + USXGB + RF',            'ct_xg_rf_results.csv',     '#2ca02c',
     lambda row: _val(row, 'ct_stat_time_s') + _val(row, 'xg_fit_time_s') + _val(row, 'rf_fit_time_s')),
    ('ST Sum + USXGB + RF + ADABoost', 'ct_xg_rf_ada_results.csv', '#ff7f0e',
     lambda row: _val(row, 'ct_stat_time_s') + _val(row, 'xg_fit_time_s') + _val(row, 'rf_fit_time_s') + _val(row, 'ada_fit_time_s')),
]

XGB_BASE_FILES = ['xgb_comparison_results.csv', 'xgb_comparison_result.csv', 'xg_only_results.csv']
XGB_BASE_NAME  = 'USXGB baseline'
XGB_BASE_COLOR = '#9467bd'

def _load_time_series_from_file(name, fname, color, time_fn):
    fpath = os.path.join(base_dir, fname)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    if 'Round' not in df.columns:
        df['Round'] = range(len(df))
    try:
        total_train = _pick_total_train_cols(df)
    except KeyError:
        return None
    df['Time'] = df.apply(lambda row: time_fn(row), axis=1)
    df['Name'] = name
    df['Color'] = color
    df['TotalTrain'] = total_train
    return df[['Name','Color','Round','TotalTrain','Time']].sort_values('TotalTrain')

series_time = []
for name, fname, color, fn in TIME_CONFIGS:
    s = _load_time_series_from_file(name, fname, color, fn)
    if s is not None and not s.empty:
        series_time.append(s)

# baseline
for base_fname in XGB_BASE_FILES:
    fpath = os.path.join(base_dir, base_fname)
    if not os.path.exists(fpath):
        continue
    dfb = pd.read_csv(fpath)
    if 'Round' not in dfb.columns:
        dfb['Round'] = range(len(dfb))
    try:
        total_train = _pick_total_train_cols(dfb)
    except KeyError:
        total_train = pd.Series(range(len(dfb)), index=dfb.index)
    time_col = None
    if 'fit_time_s' in dfb.columns:
        time_col = dfb['fit_time_s']
    elif 'xg_fit_time_s' in dfb.columns:
        time_col = dfb['xg_fit_time_s']
    if time_col is None:
        continue
    dfb['Name'] = XGB_BASE_NAME
    dfb['Color'] = XGB_BASE_COLOR
    dfb['Time'] = time_col.fillna(0.0)
    dfb['TotalTrain'] = total_train
    series_time.append(dfb[['Name','Color','Round','TotalTrain','Time']])
    break

# ========= 三個區間 =========
ranges = [
    {'mask': lambda x: x < 10_000,                     'name': 'under_10k',   'title': 'Under 10,000 Samples'},
    {'mask': lambda x: (x >= 10_000) & (x < 100_000), 'name': '10k_to_100k', 'title': '10,000 to 100,000 Samples'},
    {'mask': lambda x: x >= 100_000,                  'name': 'over_100k',   'title': 'Over 100,000 Samples'},
]

# ========= AUROC / Balanced Accuracy =========
for r in ranges:
    # AUROC
    plt.figure(figsize=(10, 6))
    for label, color, df in series:
        sub = df[r['mask'](df['TotalTrain'])]
        if sub.empty: continue
        plt.plot(sub['TotalTrain'], sub['AUROC'], label=label, color=color,
                 linestyle=linestyles.get(label,'solid'), linewidth=2)
    plt.title(f"AUROC Comparison ({r['title']})", fontsize=14)
    plt.xlabel('Training Sample Count', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"AUROC_Comparison_{r['name']}.png"), dpi=300)
    plt.close()

    # Balanced Accuracy
    plt.figure(figsize=(10, 6))
    for label, color, df in series:
        sub = df[r['mask'](df['TotalTrain'])]
        if sub.empty: continue
        plt.plot(sub['TotalTrain'], sub['BalancedAcc'], label=label, color=color,
                 linestyle=linestyles.get(label,'solid'), linewidth=2)
    plt.title(f"Balanced Accuracy Comparison ({r['title']})", fontsize=14)
    plt.xlabel('Training Sample Count', fontsize=12)
    plt.ylabel('Balanced Accuracy = (TNR + TPR) / 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"Balanced_Accuracy_Comparison_{r['name']}.png"), dpi=300)
    plt.close()

# ========= 時間圖（每10 Round 平均） =========
if series_time:
    all_time_df = pd.concat(series_time, ignore_index=True)
    linestyles_time = linestyles

    # 每10 round 分組平均
    all_time_df['RoundGroup'] = (all_time_df['Round'] // 10) * 10
    grouped = (
        all_time_df.groupby(['Name','Color','RoundGroup'])
        .agg({'TotalTrain':'mean','Time':'mean'})
        .reset_index()
        .sort_values('TotalTrain')
    )

    for r in ranges:
        plt.figure(figsize=(10,6))
        for (name, color), g in grouped.groupby(['Name','Color']):
            sub = g[r['mask'](g['TotalTrain'])]
            if sub.empty: continue
            plt.plot(sub['TotalTrain'], sub['Time'],
                     label=f"{name} (avg per 10 rounds)",
                     color=color,
                     linestyle=linestyles_time.get(name,'solid'),
                     linewidth=2)
        plt.title(f"Training Time vs. Sample Count ({r['title']})", fontsize=14)
        plt.xlabel('Training Sample Count', fontsize=12)
        plt.ylabel('Average Fit Time per 10 Rounds (s)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Time_vs_Samples_{r['name']}_10roundavg.png"), dpi=300)
        plt.close()

print(f"✅ 圖片已輸出到：{output_dir}")
