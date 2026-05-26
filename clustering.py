# =============================================================================
# clustering.py - Phân cụm khách hàng bằng KMeans
# =============================================================================
# Mô tả: Sử dụng thuật toán KMeans để phân nhóm khách hàng
#         - Elbow Method: tìm số cụm tối ưu
#         - KMeans Clustering: phân cụm
#         - Phân tích nhóm: gán tên mô tả cho từng cụm
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
import warnings
import sys

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


def get_valid_k_range(X_scaled, min_k=2, max_k=10):
    """Tra ve khoang K hop le cho KMeans va silhouette."""
    n_samples = len(X_scaled)
    if n_samples < 3:
        raise ValueError("Cần ít nhất 3 dòng dữ liệu để đánh giá KMeans bằng Silhouette.")
    return range(min_k, min(max_k, n_samples - 1) + 1)


def elbow_method(X_scaled, k_range=range(2, 11)):
    """
    Phương pháp Elbow để tìm số cụm K tối ưu.

    Chạy KMeans với K = 2..10, tính Inertia và Silhouette Score.
    K tốt nhất = K có Silhouette Score cao nhất.
    """
    print("\n📊 Elbow Method:")
    print("-" * 40)

    k_range = list(k_range)
    max_valid_k = len(X_scaled) - 1
    k_range = [k for k in k_range if 2 <= k <= max_valid_k]
    if not k_range:
        raise ValueError("Không có giá trị K hợp lệ. Cần ít nhất 3 dòng dữ liệu.")

    inertias = []
    sil_scores = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        sil = silhouette_score(X_scaled, kmeans.labels_)
        sil_scores.append(sil)
        print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil:.4f}")

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\n   🏆 K tốt nhất: K = {best_k}")

    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': sil_scores,
        'best_k': best_k
    }


def kmeans_clustering(X_scaled, n_clusters=4):
    """
    Thuật toán KMeans Clustering.

    1. Chọn ngẫu nhiên K tâm cụm
    2. Gán mỗi điểm vào cụm có tâm gần nhất
    3. Cập nhật tâm cụm = trung bình các điểm
    4. Lặp lại cho đến khi hội tụ
    """
    print(f"\n🔵 KMeans Clustering (K={n_clusters}):")
    print("-" * 40)

    if n_clusters < 2 or n_clusters >= len(X_scaled):
        raise ValueError(
            f"K={n_clusters} không hợp lệ với {len(X_scaled)} dòng dữ liệu. "
            "K phải >= 2 và nhỏ hơn số dòng dữ liệu."
        )

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=300)
    labels = model.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    print(f"   - Silhouette Score: {sil_score:.4f}")
    print(f"   - Davies-Bouldin Index: {db_score:.4f}")
    print(f"   - Calinski-Harabasz Score: {ch_score:.2f}")
    print(f"   - Inertia: {model.inertia_:.2f}")

    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"   - Cụm {cluster}: {count} khách ({count/len(labels)*100:.1f}%)")

    return {
        'labels': labels,
        'model': model,
        'silhouette': sil_score,
        'davies_bouldin': db_score,
        'calinski_harabasz': ch_score,
        'inertia': model.inertia_
    }


def analyze_clusters(df, labels):
    """
    Gán tên mô tả cho từng cụm:
    - 💎 Khách VIP: thu nhập cao, chi tiêu cao
    - ⭐ Khách tiềm năng: thu nhập & chi tiêu trung bình-khá
    - 🛒 Khách ít mua hàng: tần suất mua thấp
    - 💰 Khách chi tiêu thấp: điểm chi tiêu thấp
    """
    df_result = df.copy()
    df_result['Cluster'] = labels

    num_cols = [c for c in ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency'] if c in df_result.columns]
    cluster_stats = df_result.groupby('Cluster')[num_cols].mean()

    inc_med = df_result['Income'].median() if 'Income' in df_result.columns else 0
    spe_med = df_result['SpendingScore'].median() if 'SpendingScore' in df_result.columns else 0
    freq_med = df_result['PurchaseFrequency'].median() if 'PurchaseFrequency' in df_result.columns else 0

    cluster_names = {}
    for cid in sorted(df_result['Cluster'].unique()):
        s = cluster_stats.loc[cid]
        inc = s.get('Income', 0)
        spe = s.get('SpendingScore', 0)
        freq = s.get('PurchaseFrequency', 0)

        if inc > inc_med * 1.2 and spe > spe_med * 1.2:
            cluster_names[cid] = "💎 Khách VIP"
        elif inc > inc_med * 0.8 and spe > spe_med * 0.8:
            cluster_names[cid] = "⭐ Khách tiềm năng"
        elif freq < freq_med * 0.7:
            cluster_names[cid] = "🛒 Khách ít mua hàng"
        else:
            cluster_names[cid] = "💰 Khách chi tiêu thấp"

    df_result['ClusterName'] = df_result['Cluster'].map(cluster_names)
    return df_result


if __name__ == '__main__':
    import os
    from preprocessing import load_data, preprocess_pipeline

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers_kaggle.csv')
    if not os.path.exists(data_path):
        print("⚠️ Chưa có file customers_kaggle.csv! Hãy đặt file vào thư mục data/")
    else:
        df = load_data(data_path)
        result = preprocess_pipeline(df)
        elbow_result = elbow_method(result['X_scaled'], get_valid_k_range(result['X_scaled']))
        km_result = kmeans_clustering(result['X_scaled'], n_clusters=elbow_result['best_k'])
        df_clustered = analyze_clusters(result['df_cleaned'], km_result['labels'])
