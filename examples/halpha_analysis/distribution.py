# distribution_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from typing import Union
from unyt import unyt_array
UNYT_AVAILABLE = True


def analyze_and_plot_distribution(
    n_h_data: Union[np.ndarray, unyt_array],
    col_den_data: Union[np.ndarray, unyt_array],
    output_prefix: str = "data_distribution"
):
    """
    分析並繪製 volume number density (nH) 和 column density (N) 的數據分佈。

    這個函式會執行以下操作：
    1. 計算並印出 nH 和 N 的描述性統計數據。
    2. 繪製並儲存兩個數據集的一維直方圖 (log-log scale)。
    3. 繪製並儲存 nH 和 N 的二維聯合分佈圖 (log-log scale)。

    Args:
        n_h_data (np.ndarray or unyt.unyt_array): 包含 H 核體密度 (volume number density) 的3D數據陣列。
                                                 單位應為 cm^-3。
        col_den_data (np.ndarray or unyt.unyt_array): 包含 H 核柱密度 (column density) 的3D數據陣列。
                                                      單位應為 cm^-2。
        output_prefix (str, optional): 儲存的圖片檔案名前綴。預設為 "data_distribution"。
    """
    print("--- Starting Distribution Analysis ---")

    # --- 數據準備 ---
    # 處理 unyt_array，如果存在的話
    if UNYT_AVAILABLE and isinstance(n_h_data, unyt_array):
        n_h_flat = n_h_data.in_cgs().value.flatten()
    else:
        n_h_flat = n_h_data.flatten()
        
    if UNYT_AVAILABLE and isinstance(col_den_data, unyt_array):
        col_den_flat = col_den_data.in_cgs().value.flatten()
    else:
        col_den_flat = col_den_data.flatten()

    # 過濾掉非正數值，避免 log scale 出錯
    n_h_flat = n_h_flat[n_h_flat > 0]
    col_den_flat = col_den_flat[col_den_flat > 0]
    
    if len(n_h_flat) == 0 or len(col_den_flat) == 0:
        print("Error: Data arrays are empty or contain no positive values.")
        return

    # --- 1. 計算並印出描述性統計數據 ---
    print("\n--- n_H Statistics (cm^-3) ---")
    print(f"Min: {np.min(n_h_flat):.2e}")
    print(f"Max: {np.max(n_h_flat):.2e}")
    print(f"Mean: {np.mean(n_h_flat):.2e}")
    print(f"Median: {np.median(n_h_flat):.2e}")
    print(f"1st Percentile: {np.percentile(n_h_flat, 1):.2e}")
    print(f"99th Percentile: {np.percentile(n_h_flat, 99):.2e}")

    print("\n--- Column Density Statistics (cm^-2) ---")
    print(f"Min: {np.min(col_den_flat):.2e}")
    print(f"Max: {np.max(col_den_flat):.2e}")
    print(f"Mean: {np.mean(col_den_flat):.2e}")
    print(f"Median: {np.median(col_den_flat):.2e}")
    print(f"1st Percentile: {np.percentile(col_den_flat, 1):.2e}")
    print(f"99th Percentile: {np.percentile(col_den_flat, 99):.2e}")

    # --- 2. 繪製一維直方圖 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    log_bins_nh = np.logspace(np.log10(n_h_flat.min()), np.log10(n_h_flat.max()), 50)
    ax1.hist(n_h_flat, bins=log_bins_nh, color='skyblue', edgecolor='black')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_title('1D Histogram of n_H')
    ax1.set_xlabel('n_H (cm$^{-3}$)')
    ax1.set_ylabel('Frequency')

    log_bins_n = np.logspace(np.log10(col_den_flat.min()), np.log10(col_den_flat.max()), 50)
    ax2.hist(col_den_flat, bins=log_bins_n, color='salmon', edgecolor='black')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_title('1D Histogram of Column Density')
    ax2.set_xlabel('N_H (cm$^{-2}$)')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    hist1d_filename = f"{output_prefix}_1d_histograms.png"
    plt.savefig(hist1d_filename, dpi=300)
    print(f"\nSaved 1D histograms to: {hist1d_filename}")
    plt.show()

    # --- 3. 繪製二維聯合分佈圖 ---
    plt.figure(figsize=(8, 8))
    
    plt.hist2d(n_h_flat, col_den_flat,
               bins=[log_bins_nh, log_bins_n],
               norm=LogNorm(), cmap='viridis')

    plt.xscale('log')
    plt.yscale('log')
    plt.title('Joint Distribution of n_H and Column Density')
    plt.xlabel('n_H (cm$^{-3}$)')
    plt.ylabel('N_H (cm$^{-2}$)')
    cbar = plt.colorbar()
    cbar.set_label('Counts in bin')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    
    hist2d_filename = f"{output_prefix}_2d_histogram.png"
    plt.savefig(hist2d_filename, dpi=300)
    print(f"Saved 2D histogram to: {hist2d_filename}")
    plt.show()
    
    print("\n--- Distribution Analysis Complete ---")

if __name__ == '__main__':
    # --- 這是一個如何使用此模組的範例 ---
    # 當您直接執行 `python distribution_analyzer.py` 時，
    # 下面的程式碼會被執行，用來測試函式功能。
    
    print(">>> Running a test example...")
    
    # 產生一些符合天文數據分佈特性的假數據
    np.random.seed(42)
    sample_n_h = 10**np.random.normal(loc=2, scale=1.5, size=(100, 100, 100))
    sample_col_den = 10**np.random.normal(loc=21, scale=1.0, size=(100, 100, 100))
    
    # 呼叫主函式
    analyze_and_plot_distribution(sample_n_h, sample_col_den, output_prefix="sample_data")