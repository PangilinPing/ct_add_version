# -*- coding: utf-8 -*-
import os
import time
import pandas as pd
from typing import Dict, Tuple

def recompute_compute(data: pd.DataFrame, feature_count_dir: str = 'feature_count') -> Tuple[Dict[str, pd.DataFrame], float]:
    """
    回傳 (tables, compute_secs)
      - tables: {column -> DataFrame(index='feature_value', columns=['benign_count','full_count'])}
      - compute_secs: 只計算統計運算時間（value_counts/concat/fillna/add/型別轉換），
                      不包含任何 I/O（read_csv、to_csv、os.path.exists 等）
    """
    # 抓 benign 索引並去除 label 欄
    benign_idx = data.index[data['label'] == 0]
    X = data.drop(columns=['label'])

    tables: Dict[str, pd.DataFrame] = {}
    compute_secs = 0.0

    for column in X.columns:
        # --- 計算目前批次的 value_counts（計入時間） ---
        t0 = time.perf_counter()
        cur = pd.concat(
            [
                X.loc[benign_idx, column].value_counts(sort=False),
                X[column].value_counts(sort=False)
            ],
            axis=1, keys=['benign_count', 'full_count']
        ).fillna(0)
        cur.index.name = 'feature_value'
        compute_secs += (time.perf_counter() - t0)

        # --- 讀舊檔（若存在）：不計時 ---
        prev = None
        prev_path = os.path.join(feature_count_dir, f'{column}.csv')
        if os.path.exists(prev_path):  # 這段包含 exists 與 read，都不計時
            prev = pd.read_csv(prev_path, low_memory=False, index_col='feature_value')

        # --- 合併舊表與型別轉換（計入時間） ---
        t1 = time.perf_counter()
        merged = prev.add(cur, fill_value=0) if prev is not None else cur

        # 將計數欄位盡量轉成 int64（加總後可能成為 float）
        for c in ('benign_count', 'full_count'):
            if c in merged.columns:
                # 先轉 float 再四捨五入轉 int，避免混雜型別
                merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0).round().astype('int64')

        compute_secs += (time.perf_counter() - t1)

        tables[column] = merged

    return tables, compute_secs


def recompute_save(tables: Dict[str, pd.DataFrame], feature_count_dir: str = 'feature_count') -> float:
    """
    將 tables 一次性寫入 CSV。回傳 write_secs（只含寫檔 I/O 時間）。
    """
    os.makedirs(feature_count_dir, exist_ok=True)
    t0 = time.perf_counter()
    for column, df in tables.items():
        out_path = os.path.join(feature_count_dir, f'{column}.csv')
        df.to_csv(out_path)
    write_secs = time.perf_counter() - t0
    return write_secs


def recompute(data: pd.DataFrame, feature_count_dir: str = 'feature_count', write: bool = False):
    """
    - write=False：只做計算，回傳 (tables, compute_secs)
    - write=True ：計算後寫檔，回傳 (tables, compute_secs, write_secs)
    ※ compute_secs 不含任何 I/O（含 read_csv / to_csv / os.path.exists）。
    """
    tables, compute_secs = recompute_compute(data, feature_count_dir)
    write_secs = recompute_save(tables, feature_count_dir)
    return tables, compute_secs
