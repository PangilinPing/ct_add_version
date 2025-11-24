from collections import Counter  # 可移除，這版未使用
import pandas as pd
import numpy as np              # 可移除，這版未使用
import os
import json                     # 可移除，這版未使用
import time

# ct值 = p值 - 0.5  （僅註解，與本函式無關）
def statistic(data, out_dir: str = '../feature_count'):
    """
    回傳 compute_secs（純統計計算時間，排除所有 I/O）
    - 統計：對每個欄位計算 benign/full 的 value_counts，合併、補 0、設索引名
    - I/O：建立資料夾、逐欄寫 CSV（這些不計時）
    """
    # 準備輸出資料夾（不計時）
    os.makedirs(out_dir, exist_ok=True)

    # 取得 benign 索引並去除 label 欄
    benign_idx = data.index[data['label'] == 0]
    X = data.drop(columns=['label'])

    compute_secs = 0.0

    for column in X.columns:
        # ---- 純計算（計時）----
        t0 = time.perf_counter()
        ct_value = pd.concat(
            [
                X.loc[benign_idx, column].value_counts(sort=False),
                X[column].value_counts(sort=False)
            ],
            axis=1, keys=['benign_count', 'full_count']
        ).fillna(0)
        ct_value.index.name = 'feature_value'
        compute_secs += (time.perf_counter() - t0)

        # ---- 寫檔（不計時）----
        ct_value.to_csv(os.path.join(out_dir, f'{column}.csv'))

    return compute_secs
