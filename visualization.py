# =============================================================================
# visualization.py - Trực quan hóa dữ liệu
# =============================================================================
# Mô tả: Tạo 3 loại biểu đồ đơn giản:
#         1. Histogram - Phân phối dữ liệu
#         2. Heatmap - Ma trận tương quan
#         3. Scatter Plot - Biểu đồ phân tán
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Cấu hình
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif'],
    'axes.unicode_minus': False,
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FAFBFC',
    'axes.edgecolor':   '#D1D5DB',
    'axes.labelcolor':  '#374151',
    'axes.grid':        True,
    'grid.color':       '#E5E7EB',
    'grid.alpha':       0.6,
    'grid.linewidth':   0.5,
    'xtick.color':      '#6B7280',
    'ytick.color':      '#6B7280',
})
sns.set_style("whitegrid")

CLUSTER_COLORS = ['#2563EB', '#7C3AED', '#059669', '#D97706',
                  '#DC2626', '#0891B2', '#4F46E5', '#CA8A04']
HIST_COLORS = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B']


def plot_histograms(df, figsize=(14, 5)):
    """
    Vẽ histogram cho các cột số.
    Histogram cho thấy phân phối dữ liệu của từng đặc trưng.
    """
    num_cols = [c for c in ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency'] if c in df.columns]
    n = len(num_cols)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    colors = HIST_COLORS

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=25, color=colors[i], edgecolor='white', alpha=0.85)
        axes[i].set_title(f'Phan phoi {col}', fontweight='bold', fontsize=11)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('So luong')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.2, label=f'Mean: {df[col].mean():.1f}')
        axes[i].legend(fontsize=8)

    plt.suptitle('HISTOGRAM - Phan phoi cac dac trung', fontsize=14, fontweight='bold', y=1.03)
    plt.tight_layout()
    return fig


def plot_heatmap(df, figsize=(8, 6)):
    """
    Vẽ heatmap ma trận tương quan.
    Giá trị gần 1: tương quan thuận mạnh
    Giá trị gần -1: tương quan nghịch mạnh
    Giá trị gần 0: không tương quan
    """
    num_cols = [c for c in ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency'] if c in df.columns]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, ax=ax, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.8})
    ax.set_title('HEATMAP - Ma tran tuong quan', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    return fig


def plot_scatter(df, x_col='Income', y_col='SpendingScore', hue_col=None, figsize=(9, 6)):
    """
    Vẽ scatter plot giữa 2 biến.
    Nếu có hue_col (ví dụ: 'Cluster') thì tô màu theo nhóm.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col in df.columns:
        palette = CLUSTER_COLORS
        for i, cluster in enumerate(sorted(df[hue_col].unique())):
            subset = df[df[hue_col] == cluster]
            ax.scatter(subset[x_col], subset[y_col],
                       c=palette[i % len(palette)], label=f'Cum {cluster}',
                       s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        ax.legend(title=hue_col, fontsize=9)
    else:
        ax.scatter(df[x_col], df[y_col], c='#4ECDC4', s=50, alpha=0.6, edgecolors='white')

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'SCATTER PLOT: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_elbow(elbow_result, selected_k=None, figsize=(13, 5)):
    """
    Vẽ biểu đồ Elbow Method và Silhouette Score.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    k_values = elbow_result['k_values']
    inertias = elbow_result['inertias']
    sil_scores = elbow_result['silhouette_scores']
    show_k = selected_k or elbow_result['best_k']

    # Elbow Chart
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=show_k, color='red', linestyle='--', label=f'K={show_k}')
    ax1.set_xlabel('So cum (K)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method', fontweight='bold', fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Silhouette Chart
    bar_colors = ['#2563EB' if k == show_k else '#BFDBFE' for k in k_values]
    ax2.bar(k_values, sil_scores, color=bar_colors, edgecolor='white')
    ax2.set_xlabel('So cum (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score theo K', fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers_kaggle.csv')
    if not os.path.exists(data_path):
        print("⚠️ Chưa có file customers_kaggle.csv! Hãy đặt file vào thư mục data/")
    else:
        from preprocessing import load_data, preprocess_pipeline
        from clustering import kmeans_clustering, elbow_method

        df = load_data(data_path)
        result = preprocess_pipeline(df)

        plot_histograms(result['df_cleaned'])
        plot_heatmap(result['df_cleaned'])

        km = kmeans_clustering(result['X_scaled'], 4)
        result['df_cleaned']['Cluster'] = km['labels']
        plot_scatter(result['df_cleaned'], 'Income', 'SpendingScore', 'Cluster')

        plt.show()
        print("✅ Đã tạo biểu đồ!")
