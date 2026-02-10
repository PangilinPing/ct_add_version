# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "hBDSCAN_train.csv"
LABEL_COL = "label"
OUTPUT_DIR = "cluster_label_ratio_plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for col in df.columns:
    if col == LABEL_COL:
        continue
    if not np.issubdtype(df[col].dtype, np.number):
        continue

    uniq = df[col].dropna().unique()
    if len(uniq) > 100:
        continue

    # =====================================
    # 計算每個 cluster 內 label 的「數量」
    # =====================================
    # table: index = cluster, columns = label (0/1), values = count
    table = (
        df
        .groupby(col)[LABEL_COL]
        .value_counts()
        .unstack(fill_value=0)
        .sort_index()
    )

    # 補齊 label 0 / 1
    for lb in [0, 1]:
        if lb not in table.columns:
            table[lb] = 0

    clusters = table.index.astype(str)
    count_0 = table[0].values
    count_1 = table[1].values
    total_count = count_0 + count_1

    # =====================================
    # 繪圖（stacked count bar）
    # =====================================
    x = np.arange(len(clusters))

    plt.figure(figsize=(max(6, len(clusters) * 0.3), 4))

    plt.bar(x, count_0, color="green", label="label = 0")
    plt.bar(x, count_1, bottom=count_0, color="red", label="label = 1")

    plt.xticks(x, clusters, rotation=90)
    plt.xlabel("Cluster ID")
    plt.ylabel("Sample Count")
    plt.title(f"Label Count per Cluster - {col}")
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{col}_label_count.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Saved: {out_path}")
