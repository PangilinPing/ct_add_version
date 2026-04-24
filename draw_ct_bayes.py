import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages

# 來源與輸出目錄
base_dir = 'NB15_ct_rf_vs_bayes_mlp_result'
output_dir = os.path.join(base_dir, 'images')
os.makedirs(output_dir, exist_ok=True)

# PDF 輸出路徑
pdf_output_path = os.path.join(base_dir, 'all_plots_ct_vs_bayes.pdf')

# ================= 只保留 CT 與 Bayes =================
candidates = [
    ('ct_only_results.csv',          'CT',    '#1f77b4'),
    ('bayes_comparison_results.csv', 'Bayes', '#9467bd'),
]

# ================= 線條樣式設定 =================
linestyles = {
    'CT': 'dashed',
    'Bayes': 'solid',
}


def _pick_total_train_cols(df: pd.DataFrame) -> pd.Series:
    required_cols = {'train_0', 'train_1'}
    if not required_cols.issubset(df.columns):
        raise KeyError("缺少 train_0 與 train_1 欄位，無法計算訓練樣本總數。")
    return df['train_0'].astype(float) + df['train_1'].astype(float)


def _val(row: pd.Series, key: str, default: float = 0.0) -> float:
    if key in row and pd.notna(row[key]):
        try:
            return float(row[key])
        except Exception:
            return default
    return default


# =============== 讀 AUROC / Accuracy / BalancedAcc 資料（不做 per10round） ===============
series = []

for fname, label, color in candidates:
    fpath = os.path.join(base_dir, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] file not found: {fpath}")
        continue

    df = pd.read_csv(fpath)

    need_cols = {'AUROC', 'Accuracy', 'TNR', 'TPR', 'train_0', 'train_1'}
    if not need_cols.issubset(df.columns):
        print(f"[WARN] {fname} 缺少必要欄位，跳過")
        continue

    if 'Round' not in df.columns:
        df['Round'] = range(len(df))

    df['TotalTrain'] = _pick_total_train_cols(df)
    df['BalancedAcc'] = (df['TNR'] + df['TPR']) / 2.0
    df = df.sort_values('TotalTrain').reset_index(drop=True)

    print(f"[INFO] {label}: rows={len(df)}, TotalTrain min={df['TotalTrain'].min()}, max={df['TotalTrain'].max()}")
    series.append((label, color, df[['Round', 'TotalTrain', 'AUROC', 'Accuracy', 'BalancedAcc']]))


# =============== 時間圖資料設定（只有時間圖做 per10round） ===============
TIME_CONFIGS = [
    ('CT', 'ct_only_results.csv', '#1f77b4',
     lambda row: _val(row, 'ct_total_time_s')),

    ('Bayes', 'bayes_comparison_results.csv', '#9467bd',
     lambda row: _val(row, 'fit_time_s')),
]


def _load_time_series_from_file(name, fname, color, time_fn):
    fpath = os.path.join(base_dir, fname)
    if not os.path.exists(fpath):
        print(f"[WARN] time file not found: {fpath}")
        return None

    df = pd.read_csv(fpath)

    if 'Round' not in df.columns:
        df['Round'] = range(len(df))

    required_cols = {'train_0', 'train_1'}
    if not required_cols.issubset(df.columns):
        print(f"[WARN] {fname} 缺少 train_0/train_1，跳過時間圖")
        return None

    df['Time'] = df.apply(lambda row: time_fn(row), axis=1)
    df['Name'] = name
    df['Color'] = color
    df['TotalTrain'] = _pick_total_train_cols(df)

    # ===== 只有時間圖做每10輪平均 =====
    df['RoundGroup'] = (df['Round'] // 10) * 10
    df = (
        df.groupby(['Name', 'Color', 'RoundGroup'], as_index=False)
        .agg({
            'TotalTrain': 'mean',
            'Time': 'mean'
        })
        .sort_values('TotalTrain')
        .reset_index(drop=True)
    )

    print(f"[INFO] Time-{name}: grouped_rows={len(df)}, TotalTrain min={df['TotalTrain'].min()}, max={df['TotalTrain'].max()}")

    return df[['Name', 'Color', 'RoundGroup', 'TotalTrain', 'Time']]


series_time = []
for name, fname, color, fn in TIME_CONFIGS:
    s = _load_time_series_from_file(name, fname, color, fn)
    if s is not None and not s.empty:
        series_time.append(s)


# ========= 三個區間 =========
ranges = [
    {'mask': lambda x: x < 10_000,                     'name': 'under_10k',   'title': 'Under 10,000 Samples'},
    {'mask': lambda x: (x >= 10_000) & (x < 100_000), 'name': '10k_to_100k', 'title': '10,000 to 100,000 Samples'},
    {'mask': lambda x: x >= 100_000,                  'name': 'over_100k',   'title': 'Over 100,000 Samples'},
]

generated_images = []


def save_and_track(fig, path):
    fig.savefig(path, dpi=300)
    generated_images.append(path)
    plt.close(fig)


# ========= AUROC / Accuracy / Balanced Accuracy（原始每輪） =========
for r in ranges:
    print(f"\n[INFO] ===== Range: {r['name']} =====")

    # AUROC
    fig = plt.figure(figsize=(10, 6))
    has_data = False
    for label, color, df in series:
        sub = df[r['mask'](df['TotalTrain'])]
        print(f"[INFO] {label} -> {r['name']} rows: {len(sub)}")
        if sub.empty:
            continue
        has_data = True
        plt.plot(
            sub['TotalTrain'],
            sub['AUROC'],
            label=label,
            color=color,
            linestyle=linestyles.get(label, 'solid'),
            linewidth=2
        )
    plt.title(f"AUROC Comparison ({r['title']})", fontsize=14)
    plt.xlabel('Training Sample Count', fontsize=12)
    plt.ylabel('AUROC', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    if has_data:
        plt.legend(fontsize=10)
    plt.tight_layout()
    save_and_track(fig, os.path.join(output_dir, f"AUROC_CT_vs_Bayes_{r['name']}.png"))

    # Accuracy
    fig = plt.figure(figsize=(10, 6))
    has_data = False
    for label, color, df in series:
        sub = df[r['mask'](df['TotalTrain'])]
        if sub.empty:
            continue
        has_data = True
        plt.plot(
            sub['TotalTrain'],
            sub['Accuracy'],
            label=label,
            color=color,
            linestyle=linestyles.get(label, 'solid'),
            linewidth=2
        )
    plt.title(f"Accuracy Comparison ({r['title']})", fontsize=14)
    plt.xlabel('Training Sample Count', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    if has_data:
        plt.legend(fontsize=10)
    plt.tight_layout()
    save_and_track(fig, os.path.join(output_dir, f"Accuracy_CT_vs_Bayes_{r['name']}.png"))

    # Balanced Accuracy
    fig = plt.figure(figsize=(10, 6))
    has_data = False
    for label, color, df in series:
        sub = df[r['mask'](df['TotalTrain'])]
        if sub.empty:
            continue
        has_data = True
        plt.plot(
            sub['TotalTrain'],
            sub['BalancedAcc'],
            label=label,
            color=color,
            linestyle=linestyles.get(label, 'solid'),
            linewidth=2
        )
    plt.title(f"Balanced Accuracy Comparison ({r['title']})", fontsize=14)
    plt.xlabel('Training Sample Count', fontsize=12)
    plt.ylabel('Balanced Accuracy = (TNR + TPR) / 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    if has_data:
        plt.legend(fontsize=10)
    plt.tight_layout()
    save_and_track(fig, os.path.join(output_dir, f"BalancedAcc_CT_vs_Bayes_{r['name']}.png"))


# ========= 時間圖（每10 Round 平均） =========
if series_time:
    all_time_df = pd.concat(series_time, ignore_index=True)

    for r in ranges:
        fig = plt.figure(figsize=(10, 6))
        has_data = False
        for (name, color), g in all_time_df.groupby(['Name', 'Color']):
            sub = g[r['mask'](g['TotalTrain'])]
            print(f"[INFO] Time {name} -> {r['name']} grouped rows: {len(sub)}")
            if sub.empty:
                continue
            has_data = True
            plt.plot(
                sub['TotalTrain'],
                sub['Time'],
                label=f"{name} (avg per 10 rounds)",
                color=color,
                linestyle=linestyles.get(name, 'solid'),
                linewidth=2
            )
        plt.title(f"Training Time vs. Sample Count ({r['title']}, Avg per 10 Rounds)", fontsize=14)
        plt.xlabel('Training Sample Count', fontsize=12)
        plt.ylabel('Average Fit Time per 10 Rounds (s)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        if has_data:
            plt.legend(fontsize=9)
        plt.tight_layout()
        save_and_track(fig, os.path.join(output_dir, f"Time_CT_vs_Bayes_{r['name']}_10roundavg.png"))


# ========= 將所有圖片寫入同一個 PDF =========
if generated_images:
    with PdfPages(pdf_output_path) as pdf:
        for img_path in generated_images:
            try:
                img = mpimg.imread(img_path)

                fig = plt.figure(figsize=(11.69, 8.27))  # A4 橫式
                plt.imshow(img)
                plt.axis('off')
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                print(f"[PDF] added: {img_path}")
            except Exception as e:
                print(f"[WARN] 無法加入 PDF: {img_path}, error={e}")

    print(f"✅ PDF 已輸出：{pdf_output_path}")
else:
    print("[WARN] 沒有可寫入 PDF 的圖片")

print(f"✅ 圖片已輸出到：{output_dir}")