# =============================================================================
# visualization.py - Trực quan hóa dữ liệu
# =============================================================================
# Mô tả: Tạo các biểu đồ phân tích dữ liệu khách hàng
#         bao gồm Histogram, Heatmap, Scatterplot, Boxplot, Pairplot
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Cấu hình matplotlib cho tiếng Việt
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

# Thiết lập style chung
sns.set_style("whitegrid")
sns.set_palette("husl")


# =============================================================================
# 1. HISTOGRAM - Phân phối các đặc trưng
# =============================================================================
def plot_histograms(df, figsize=(16, 10)):
    """
    Vẽ histogram cho các cột số trong dataset.

    Histogram giúp nhìn tổng quan về phân phối dữ liệu:
    - Dữ liệu có phân phối chuẩn hay lệch?
    - Có outliers không?
    - Phạm vi giá trị như thế nào?

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
        Figure chứa các histogram
    """
    numeric_cols = ['Age', 'Income', 'SpendingScore',
                    'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        ax.hist(df[col].dropna(), bins=30, color=colors[i % len(colors)],
                edgecolor='white', alpha=0.8)
        ax.set_title(f'Phan phoi {col}', fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('So luong', fontsize=10)
        ax.axvline(df[col].mean(), color='red', linestyle='--',
                   label=f'Mean: {df[col].mean():.2f}')
        ax.axvline(df[col].median(), color='blue', linestyle='--',
                   label=f'Median: {df[col].median():.2f}')
        ax.legend(fontsize=8)

    # Ẩn các subplot thừa
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('HISTOGRAM - Phan phoi cac dac trung', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# 2. HEATMAP - Ma trận tương quan
# =============================================================================
def plot_heatmap(df, figsize=(10, 8)):
    """
    Vẽ heatmap ma trận tương quan giữa các biến số.

    Heatmap tương quan cho biết:
    - Giá trị gần 1: tương quan thuận mạnh
    - Giá trị gần -1: tương quan nghịch mạnh
    - Giá trị gần 0: không tương quan

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
        Figure chứa heatmap
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Loại bỏ các cột encoded
    numeric_cols = [c for c in numeric_cols if '_Encoded' not in c]

    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)

    # Tạo mask cho nửa trên tam giác
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,           # Hiển thị giá trị
        fmt='.2f',            # Định dạng 2 chữ số thập phân
        cmap='RdYlBu_r',     # Bảng màu đẹp
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
        vmin=-1, vmax=1,
        cbar_kws={"shrink": 0.8}
    )

    ax.set_title('HEATMAP - Ma tran tuong quan', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig


# =============================================================================
# 3. SCATTERPLOT - Biểu đồ phân tán
# =============================================================================
def plot_scatter(df, x_col='Income', y_col='SpendingScore',
                 hue_col=None, figsize=(10, 7)):
    """
    Vẽ scatter plot giữa 2 biến.

    Scatter plot giúp nhìn thấy:
    - Mối quan hệ giữa 2 biến
    - Các nhóm/cụm tự nhiên
    - Outliers

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    x_col, y_col : str
        Tên cột cho trục X và Y
    hue_col : str, optional
        Cột dùng để tô màu (ví dụ: 'Cluster')
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
        Figure chứa scatter plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    if hue_col and hue_col in df.columns:
        scatter = sns.scatterplot(
            data=df, x=x_col, y=y_col, hue=hue_col,
            palette='viridis', s=50, alpha=0.7, ax=ax
        )
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(df[x_col], df[y_col], c='#4ECDC4', s=50, alpha=0.6, edgecolors='white')

    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_title(f'SCATTER PLOT: {x_col} vs {y_col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# 4. BOXPLOT - Biểu đồ hộp
# =============================================================================
def plot_boxplots(df, figsize=(16, 10)):
    """
    Vẽ boxplot cho các cột số.

    Boxplot giúp nhận diện:
    - Phân vị (Q1, Q2/median, Q3)
    - IQR (Khoảng tứ phân vị)
    - Outliers (điểm nằm ngoài 1.5*IQR)

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
        Figure chứa boxplots
    """
    numeric_cols = ['Age', 'Income', 'SpendingScore',
                    'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        bp = ax.boxplot(df[col].dropna(), patch_artist=True, vert=True)
        bp['boxes'][0].set_facecolor(colors[i % len(colors)])
        bp['boxes'][0].set_alpha(0.7)
        ax.set_title(f'Boxplot: {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col, fontsize=10)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('BOXPLOT - Phat hien Outliers', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# 5. PAIRPLOT - Biểu đồ cặp
# =============================================================================
def plot_pairplot(df, hue_col=None, figsize=None):
    """
    Vẽ pairplot cho các cặp biến số.

    Pairplot tạo ma trận biểu đồ:
    - Đường chéo: histogram/KDE của từng biến
    - Ngoài đường chéo: scatter plot của từng cặp biến
    → Giúp nhìn tổng quan mối quan hệ giữa tất cả các biến

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame chứa dữ liệu
    hue_col : str, optional
        Cột dùng để tô màu
    figsize : tuple, optional
        Kích thước figure

    Returns:
    --------
    seaborn.PairGrid
        PairGrid object
    """
    cols = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']
    cols = [c for c in cols if c in df.columns]

    plot_df = df[cols].copy()
    if hue_col and hue_col in df.columns:
        plot_df[hue_col] = df[hue_col]

    g = sns.pairplot(
        plot_df,
        hue=hue_col if hue_col and hue_col in df.columns else None,
        palette='viridis',
        diag_kind='kde',
        plot_kws={'alpha': 0.5, 's': 30},
        height=2.5
    )
    g.figure.suptitle('PAIRPLOT - Quan he giua cac bien', fontsize=16,
                  fontweight='bold', y=1.02)
    return g.figure


# =============================================================================
# 6. ELBOW & SILHOUETTE CHART
# =============================================================================
def plot_elbow(elbow_result, figsize=(14, 5)):
    """
    Vẽ biểu đồ Elbow Method và Silhouette Score.

    Parameters:
    -----------
    elbow_result : dict
        Kết quả từ hàm elbow_method()
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    k_values = elbow_result['k_values']
    inertias = elbow_result['inertias']
    sil_scores = elbow_result['silhouette_scores']
    best_k = elbow_result['best_k']

    # Elbow Chart
    ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=best_k, color='red', linestyle='--', label=f'Best K = {best_k}')
    ax1.set_xlabel('So cum (K)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Silhouette Score Chart
    colors = ['#FF6B6B' if k == best_k else '#4ECDC4' for k in k_values]
    ax2.bar(k_values, sil_scores, color=colors, edgecolor='white', alpha=0.8)
    ax2.set_xlabel('So cum (K)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score theo K', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    return fig


# =============================================================================
# 7. CLUSTER VISUALIZATION (PCA 2D)
# =============================================================================
def plot_clusters_2d(X_scaled, labels, algorithm_name='KMeans', figsize=(10, 7)):
    """
    Vẽ kết quả clustering trên không gian 2D (dùng PCA giảm chiều).

    PCA (Principal Component Analysis):
    - Giảm chiều dữ liệu từ nhiều chiều xuống 2 chiều
    - Giữ lại thông tin quan trọng nhất
    - Giúp trực quan hóa kết quả clustering

    Parameters:
    -----------
    X_scaled : np.ndarray
        Dữ liệu đã chuẩn hóa
    labels : np.ndarray
        Nhãn cụm
    algorithm_name : str
        Tên thuật toán
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
    """
    # Giảm chiều bằng PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=figsize)

    # Vẽ scatter plot với màu theo cụm
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = f'Noise' if label == -1 else f'Cum {label}'
        color = 'gray' if label == -1 else colors[i]
        alpha = 0.3 if label == -1 else 0.7

        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=[color], label=name, s=40, alpha=alpha, edgecolors='white',
                   linewidth=0.5)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'Ket qua Clustering - {algorithm_name} (PCA 2D)',
                 fontsize=14, fontweight='bold')
    ax.legend(title='Cum', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return fig


# =============================================================================
# 8. CLUSTER COMPARISON - So sánh đặc trưng các cụm
# =============================================================================
def plot_cluster_comparison(df, cluster_col='Cluster', figsize=(16, 10)):
    """
    So sánh đặc trưng trung bình giữa các cụm bằng bar chart.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame với cột cluster
    cluster_col : str
        Tên cột cluster
    figsize : tuple
        Kích thước figure

    Returns:
    --------
    matplotlib.figure.Figure
    """
    numeric_cols = ['Age', 'Income', 'SpendingScore',
                    'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays']
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # Tính trung bình theo cụm
    cluster_means = df.groupby(cluster_col)[numeric_cols].mean()

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    colors = plt.cm.Set2(np.linspace(0, 1, len(cluster_means)))

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        bars = ax.bar(
            cluster_means.index.astype(str),
            cluster_means[col],
            color=colors,
            edgecolor='white',
            alpha=0.8
        )
        ax.set_title(f'{col}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Cum', fontsize=10)
        ax.set_ylabel('Trung binh', fontsize=10)

        # Thêm giá trị trên mỗi bar
        for bar, val in zip(bars, cluster_means[col]):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('SO SANH DAC TRUNG GIUA CAC CUM',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN - Chạy trực tiếp để test visualization
# =============================================================================
if __name__ == '__main__':
    import os

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers.csv')

    if not os.path.exists(data_path):
        print("⚠️ Chưa có file customers.csv! Hãy chạy generate_data.py trước.")
    else:
        from preprocessing import load_data, preprocess_pipeline
        from clustering import kmeans_clustering, elbow_method

        df = load_data(data_path)
        result = preprocess_pipeline(df)

        # Test các biểu đồ
        print("📊 Tạo biểu đồ...")

        plot_histograms(result['df_cleaned'])
        plt.savefig('output/histograms.png', dpi=150, bbox_inches='tight')

        plot_heatmap(result['df_cleaned'])
        plt.savefig('output/heatmap.png', dpi=150, bbox_inches='tight')

        plot_boxplots(result['df_cleaned'])
        plt.savefig('output/boxplots.png', dpi=150, bbox_inches='tight')

        # Clustering rồi vẽ
        km_result = kmeans_clustering(result['X_scaled'], n_clusters=4)
        result['df_cleaned']['Cluster'] = km_result['labels']

        plot_scatter(result['df_cleaned'], 'Income', 'SpendingScore', 'Cluster')
        plt.savefig('output/scatter_clusters.png', dpi=150, bbox_inches='tight')

        plot_clusters_2d(result['X_scaled'], km_result['labels'], 'KMeans')
        plt.savefig('output/clusters_2d.png', dpi=150, bbox_inches='tight')

        print("✅ Đã lưu tất cả biểu đồ vào thư mục output/")
        plt.show()
