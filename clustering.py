# =============================================================================
# clustering.py - Áp dụng các thuật toán gom cụm
# =============================================================================
# Mô tả: Triển khai 3 thuật toán clustering:
#         1. KMeans Clustering
#         2. DBSCAN
#         3. Agglomerative Clustering
#         Kèm theo đánh giá mô hình (Elbow, Silhouette)
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# 1. ELBOW METHOD - Tìm số cụm tối ưu cho KMeans
# =============================================================================
def elbow_method(X_scaled, k_range=range(2, 11)):
    """
    Phương pháp Elbow để tìm số cụm (K) tối ưu.

    Ý tưởng: Chạy KMeans với nhiều giá trị K khác nhau,
    tính inertia (tổng bình phương khoảng cách trong cụm).
    Điểm "khuỷu tay" (elbow) là nơi inertia giảm chậm lại
    → K tối ưu.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Dữ liệu đã chuẩn hóa
    k_range : range
        Phạm vi giá trị K cần thử (mặc định: 2-10)

    Returns:
    --------
    dict
        - 'k_values': danh sách giá trị K
        - 'inertias': danh sách giá trị inertia tương ứng
        - 'silhouette_scores': danh sách Silhouette Score tương ứng
    """
    print("\n📊 Elbow Method:")
    print("-" * 40)

    inertias = []
    sil_scores = []

    for k in k_range:
        # Chạy KMeans với k cụm
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)

        # Lưu inertia
        inertias.append(kmeans.inertia_)

        # Tính Silhouette Score
        sil = silhouette_score(X_scaled, kmeans.labels_)
        sil_scores.append(sil)

        print(f"   K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={sil:.4f}")

    # Tìm K tốt nhất dựa trên Silhouette Score cao nhất
    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"\n   🏆 K tốt nhất (Silhouette): K = {best_k}")

    return {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': sil_scores,
        'best_k': best_k
    }


# =============================================================================
# 2. KMEANS CLUSTERING
# =============================================================================
def kmeans_clustering(X_scaled, n_clusters=4):
    """
    Thuật toán KMeans Clustering.

    Nguyên lý hoạt động:
    1. Chọn ngẫu nhiên K tâm cụm (centroids)
    2. Gán mỗi điểm vào cụm có tâm gần nhất
    3. Cập nhật tâm cụm = trung bình các điểm trong cụm
    4. Lặp lại bước 2-3 cho đến khi hội tụ

    Parameters:
    -----------
    X_scaled : np.ndarray
        Dữ liệu đã chuẩn hóa
    n_clusters : int
        Số cụm (mặc định: 4)

    Returns:
    --------
    dict
        - 'labels': nhãn cụm cho mỗi điểm dữ liệu
        - 'model': mô hình KMeans đã train
        - 'silhouette': Silhouette Score
        - 'inertia': giá trị Inertia
    """
    print(f"\n🔵 KMeans Clustering (K={n_clusters}):")
    print("-" * 40)

    # Khởi tạo và huấn luyện mô hình
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,       # Số lần khởi tạo ngẫu nhiên
        max_iter=300     # Số vòng lặp tối đa
    )
    labels = model.fit_predict(X_scaled)

    # Đánh giá
    sil_score = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    print(f"   - Số cụm: {n_clusters}")
    print(f"   - Silhouette Score: {sil_score:.4f}")
    print(f"   - Calinski-Harabasz Score: {ch_score:.2f}")
    print(f"   - Inertia: {model.inertia_:.2f}")

    # Thống kê số lượng mỗi cụm
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"   - Cụm {cluster}: {count} khách hàng ({count/len(labels)*100:.1f}%)")

    return {
        'labels': labels,
        'model': model,
        'silhouette': sil_score,
        'calinski_harabasz': ch_score,
        'inertia': model.inertia_,
        'algorithm': 'KMeans'
    }


# =============================================================================
# 3. DBSCAN CLUSTERING
# =============================================================================
def dbscan_clustering(X_scaled, eps=1.5, min_samples=5):
    """
    Thuật toán DBSCAN (Density-Based Spatial Clustering).

    Nguyên lý hoạt động:
    - Dựa trên mật độ điểm dữ liệu
    - eps: bán kính vùng lân cận
    - min_samples: số điểm tối thiểu để tạo thành cụm
    - Điểm không thuộc cụm nào → noise (nhãn = -1)

    Ưu điểm: Tự động xác định số cụm, phát hiện noise

    Parameters:
    -----------
    X_scaled : np.ndarray
        Dữ liệu đã chuẩn hóa
    eps : float
        Bán kính vùng lân cận (mặc định: 1.5)
    min_samples : int
        Số điểm tối thiểu trong vùng lân cận (mặc định: 5)

    Returns:
    --------
    dict
        Kết quả clustering
    """
    print(f"\n🟢 DBSCAN Clustering (eps={eps}, min_samples={min_samples}):")
    print("-" * 40)

    # Khởi tạo và huấn luyện
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    # Đếm số cụm (không tính noise = -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"   - Số cụm phát hiện: {n_clusters}")
    print(f"   - Số điểm noise: {n_noise} ({n_noise/len(labels)*100:.1f}%)")

    # Tính Silhouette Score (nếu có ít nhất 2 cụm và không phải tất cả noise)
    sil_score = -1
    ch_score = -1
    if n_clusters >= 2:
        # Chỉ tính trên các điểm không phải noise
        mask = labels != -1
        if mask.sum() > n_clusters:
            sil_score = silhouette_score(X_scaled[mask], labels[mask])
            ch_score = calinski_harabasz_score(X_scaled[mask], labels[mask])
            print(f"   - Silhouette Score: {sil_score:.4f}")
            print(f"   - Calinski-Harabasz Score: {ch_score:.2f}")

    # Thống kê
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        label_name = f"Cụm {cluster}" if cluster != -1 else "Noise"
        print(f"   - {label_name}: {count} khách hàng ({count/len(labels)*100:.1f}%)")

    return {
        'labels': labels,
        'model': model,
        'silhouette': sil_score,
        'calinski_harabasz': ch_score,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'algorithm': 'DBSCAN'
    }


# =============================================================================
# 4. AGGLOMERATIVE CLUSTERING
# =============================================================================
def agglomerative_clustering(X_scaled, n_clusters=4, linkage='ward'):
    """
    Thuật toán Agglomerative Clustering (Phân cụm phân cấp).

    Nguyên lý hoạt động (Bottom-up):
    1. Ban đầu mỗi điểm là một cụm riêng
    2. Gộp 2 cụm gần nhất thành 1 cụm
    3. Lặp lại bước 2 cho đến khi còn K cụm

    Linkage methods:
    - 'ward': tối thiểu tổng phương sai trong cụm
    - 'complete': khoảng cách max giữa các cụm
    - 'average': khoảng cách trung bình giữa các cụm
    - 'single': khoảng cách min giữa các cụm

    Parameters:
    -----------
    X_scaled : np.ndarray
        Dữ liệu đã chuẩn hóa
    n_clusters : int
        Số cụm (mặc định: 4)
    linkage : str
        Phương pháp liên kết (mặc định: 'ward')

    Returns:
    --------
    dict
        Kết quả clustering
    """
    print(f"\n🟡 Agglomerative Clustering (K={n_clusters}, linkage='{linkage}'):")
    print("-" * 40)

    # Khởi tạo và huấn luyện
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(X_scaled)

    # Đánh giá
    sil_score = silhouette_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)

    print(f"   - Số cụm: {n_clusters}")
    print(f"   - Linkage: {linkage}")
    print(f"   - Silhouette Score: {sil_score:.4f}")
    print(f"   - Calinski-Harabasz Score: {ch_score:.2f}")

    # Thống kê
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"   - Cụm {cluster}: {count} khách hàng ({count/len(labels)*100:.1f}%)")

    return {
        'labels': labels,
        'model': model,
        'silhouette': sil_score,
        'calinski_harabasz': ch_score,
        'algorithm': 'Agglomerative'
    }


# =============================================================================
# 5. PHÂN TÍCH NHÓM KHÁCH HÀNG
# =============================================================================
def analyze_clusters(df, labels, feature_cols=None):
    """
    Phân tích đặc điểm từng nhóm khách hàng.

    Gán nhãn mô tả cho mỗi cụm dựa trên đặc điểm:
    - Khách VIP: Thu nhập cao, chi tiêu cao
    - Khách tiềm năng: Thu nhập trung bình, chi tiêu trung bình
    - Khách tiết kiệm: Thu nhập thấp, chi tiêu thấp
    - Khách săn giảm giá: Tần suất mua cao, giá trị đơn thấp

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame gốc (trước chuẩn hóa)
    labels : np.ndarray
        Nhãn cụm từ thuật toán clustering

    Returns:
    --------
    pd.DataFrame
        DataFrame với cột 'Cluster' và 'ClusterName' được thêm vào
    """
    print("\n📋 Phân tích nhóm khách hàng:")
    print("=" * 60)

    # Thêm cột cluster vào DataFrame
    df_result = df.copy()
    df_result['Cluster'] = labels

    # Lọc bỏ noise points (DBSCAN)
    df_valid = df_result[df_result['Cluster'] != -1]

    # Tính thống kê trung bình cho từng cụm
    numeric_cols = ['Age', 'Income', 'SpendingScore',
                    'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays']
    numeric_cols = [c for c in numeric_cols if c in df_valid.columns]

    cluster_stats = df_valid.groupby('Cluster')[numeric_cols].mean()
    print("\n📊 Thống kê trung bình theo cụm:")
    print(cluster_stats.round(2))

    # Gán tên nhóm dựa trên đặc điểm
    cluster_names = {}
    for cluster_id in sorted(df_valid['Cluster'].unique()):
        stats = cluster_stats.loc[cluster_id]

        income_median = df_valid['Income'].median() if 'Income' in df_valid.columns else 0
        spending_median = df_valid['SpendingScore'].median() if 'SpendingScore' in df_valid.columns else 0

        income = stats.get('Income', 0)
        spending = stats.get('SpendingScore', 0)
        frequency = stats.get('PurchaseFrequency', 0)
        avg_order = stats.get('AverageOrderValue', 0)

        # Logic phân loại khách hàng
        if income > income_median * 1.3 and spending > spending_median * 1.2:
            name = "💎 Khách VIP"
        elif frequency > df_valid['PurchaseFrequency'].median() * 1.3 and avg_order < df_valid['AverageOrderValue'].median() * 0.8:
            name = "🏷️ Khách săn giảm giá"
        elif income > income_median * 0.8 and spending > spending_median * 0.8:
            name = "⭐ Khách tiềm năng"
        elif stats.get('LastPurchaseDays', 0) > df_valid['LastPurchaseDays'].median() * 1.5:
            name = "😴 Khách không hoạt động"
        else:
            name = "💰 Khách tiết kiệm"

        cluster_names[cluster_id] = name

    # Thêm tên nhóm vào DataFrame
    df_result['ClusterName'] = df_result['Cluster'].map(cluster_names)
    # Xử lý noise points
    df_result['ClusterName'] = df_result['ClusterName'].fillna("🔇 Noise")

    # In kết quả
    print("\n🏷️ Phân loại nhóm khách hàng:")
    print("-" * 40)
    for cluster_id, name in sorted(cluster_names.items()):
        count = (df_result['Cluster'] == cluster_id).sum()
        pct = count / len(df_result) * 100
        print(f"   Cụm {cluster_id} → {name} ({count} khách - {pct:.1f}%)")

    return df_result


# =============================================================================
# 6. SO SÁNH CÁC THUẬT TOÁN
# =============================================================================
def compare_algorithms(X_scaled, n_clusters=4):
    """
    So sánh kết quả của 3 thuật toán clustering.

    Parameters:
    -----------
    X_scaled : np.ndarray
        Dữ liệu đã chuẩn hóa
    n_clusters : int
        Số cụm cho KMeans và Agglomerative

    Returns:
    --------
    pd.DataFrame
        Bảng so sánh các thuật toán
    """
    print("\n" + "=" * 60)
    print("📊 SO SÁNH CÁC THUẬT TOÁN CLUSTERING")
    print("=" * 60)

    results = {}

    # KMeans
    results['KMeans'] = kmeans_clustering(X_scaled, n_clusters)

    # DBSCAN
    results['DBSCAN'] = dbscan_clustering(X_scaled)

    # Agglomerative
    results['Agglomerative'] = agglomerative_clustering(X_scaled, n_clusters)

    # Tạo bảng so sánh
    comparison = pd.DataFrame({
        'Thuật toán': ['KMeans', 'DBSCAN', 'Agglomerative'],
        'Silhouette Score': [
            results['KMeans']['silhouette'],
            results['DBSCAN']['silhouette'],
            results['Agglomerative']['silhouette']
        ],
        'Calinski-Harabasz': [
            results['KMeans']['calinski_harabasz'],
            results['DBSCAN']['calinski_harabasz'],
            results['Agglomerative']['calinski_harabasz']
        ]
    })

    print("\n📋 Bảng so sánh:")
    print(comparison.to_string(index=False))

    return results, comparison


# =============================================================================
# MAIN - Chạy trực tiếp để test clustering
# =============================================================================
if __name__ == '__main__':
    import os
    from preprocessing import load_data, preprocess_pipeline

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers.csv')

    if not os.path.exists(data_path):
        print("⚠️ Chưa có file customers.csv! Hãy chạy generate_data.py trước.")
    else:
        # Đọc và tiền xử lý
        df = load_data(data_path)
        result = preprocess_pipeline(df)

        # Tìm K tối ưu
        elbow_result = elbow_method(result['X_scaled'])

        # So sánh thuật toán
        algo_results, comparison = compare_algorithms(
            result['X_scaled'],
            n_clusters=elbow_result['best_k']
        )

        # Phân tích nhóm khách hàng (dùng KMeans)
        df_clustered = analyze_clusters(
            result['df_cleaned'],
            algo_results['KMeans']['labels']
        )
