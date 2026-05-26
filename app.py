# =============================================================================
# app.py - Dashboard Streamlit đơn giản
# =============================================================================
# Mô tả: Giao diện web phân cụm khách hàng bằng KMeans
#         - Upload CSV hoặc dùng dataset Kaggle mặc định
#         - Chọn số cụm K
#         - Hiển thị biểu đồ: Histogram, Heatmap, Scatter Plot
#         - Hiển thị bảng dữ liệu và mô tả nhóm khách hàng
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from clustering import analyze_clusters, elbow_method, get_valid_k_range, kmeans_clustering
from preprocessing import preprocess_pipeline

warnings.filterwarnings('ignore')

# ======================== CẤU HÌNH TRANG ========================
st.set_page_config(
    page_title="Phân Cụm Khách Hàng - KMeans Nhóm 08",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p { margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1rem; }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }
    .metric-value {
        font-size: 2rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 0.3rem; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea22, transparent);
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600; font-size: 1.1rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8eaf6 0%, #f3e5f5 50%, #ede7f6 100%);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #4a148c !important; }

    /* Cluster info */
    .cluster-info {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)


# ======================== HÀM HỖ TRỢ ========================
@st.cache_data
def load_default_data():
    """Tải dataset mặc định từ thư mục data/"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers_kaggle.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("## 🎛️ Bảng Điều Khiển")
    st.markdown("---")

    # Upload hoặc dùng mặc định
    st.markdown("### 📁 Dữ liệu")
    upload = st.file_uploader("Upload file CSV", type=['csv'])
    use_default = st.checkbox("Dùng dataset Kaggle mặc định", value=True)

    st.markdown("---")
    st.markdown("### ⚙️ Cài đặt KMeans")
    k = st.slider("Số cụm (K)", 2, 10, 8)

    st.markdown("---")
    st.markdown("### 📊 Biểu đồ")
    show_hist = st.checkbox("Histogram", value=True)
    show_heatmap = st.checkbox("Heatmap", value=True)

    run_btn = st.button("🚀 Chạy Phân Cụm", width="stretch", type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; opacity:0.6; font-size:0.8rem;'>
        Nhóm 08 — KPDL<br>© 2026
    </div>
    """, unsafe_allow_html=True)


# ======================== NỘI DUNG CHÍNH ========================
# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Phân Cụm Khách Hàng bằng KMeans</h1>
    <p>Đồ án môn Khai Phá Dữ Liệu — Nhóm 08</p>
</div>
""", unsafe_allow_html=True)

# Load data
df_raw = None
if upload is not None:
    df_raw = pd.read_csv(upload)
    st.success(f"✅ Đã tải file: {upload.name} ({len(df_raw)} dòng)")
elif use_default:
    df_raw = load_default_data()
    if df_raw is not None:
        st.info(f"📂 Đang dùng dataset Kaggle mặc định ({len(df_raw)} dòng)")
    else:
        st.warning("⚠️ Chưa có dataset mặc định. Hãy đặt file `customers_kaggle.csv` vào thư mục `data/` hoặc upload CSV.")

if df_raw is None:
    st.stop()

# Tiền xử lý
try:
    preprocess_result = preprocess_pipeline(df_raw)
except ValueError as exc:
    st.error(f"Không thể tiền xử lý dữ liệu: {exc}")
    st.info("Hãy upload file đã chuẩn hóa có các cột: CustomerID, Gender, Age, Income, SpendingScore, PurchaseFrequency.")
    st.stop()

df = preprocess_result['df_cleaned']
df_input = preprocess_result['df_input']
X_scaled = preprocess_result['X_scaled']
feature_cols = preprocess_result['feature_cols']

# ---- TỔNG QUAN DỮ LIỆU ----
st.markdown('<div class="section-header">📋 Tổng Quan Dữ Liệu</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Khách hàng</div></div>', unsafe_allow_html=True)
with c2:
    missing = df_input.isnull().sum().sum()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{missing}</div><div class="metric-label">Missing (trước xử lý)</div></div>', unsafe_allow_html=True)
with c3:
    dup = df_input.duplicated().sum()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{dup}</div><div class="metric-label">Duplicates (trước xử lý)</div></div>', unsafe_allow_html=True)

# Dữ liệu mẫu
with st.expander("🔍 Xem dữ liệu mẫu", expanded=False):
    st.dataframe(df.head(15), width="stretch")

# ---- BIỂU ĐỒ KHÁM PHÁ ----
st.markdown('<div class="section-header">📈 Khám Phá Dữ Liệu (EDA)</div>', unsafe_allow_html=True)

num_cols = feature_cols

if show_hist:
    st.markdown("#### 📊 Histogram — Phân phối dữ liệu")
    fig, axes = plt.subplots(1, len(num_cols), figsize=(14, 4))
    if len(num_cols) == 1:
        axes = [axes]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=25, color=colors[i], edgecolor='white', alpha=0.85)
        axes[i].set_title(col, fontweight='bold')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

if show_heatmap:
    st.markdown("#### 🌡️ Heatmap — Ma trận tương quan")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Ma tran tuong quan', fontweight='bold', fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ======================== PHÂN CỤM KMEANS ========================
if run_btn:
    st.markdown('<div class="section-header">🤖 Kết Quả Phân Cụm KMeans</div>', unsafe_allow_html=True)

    X = X_scaled

    try:
        k_values = list(get_valid_k_range(X))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    selected_k = min(k, max(k_values))
    if selected_k != k:
        st.warning(f"K={k} không hợp lệ với số dòng hiện tại. Đã dùng K={selected_k}.")

    # --- Elbow Method ---
    st.markdown("#### 📐 Elbow Method & Silhouette Score")
    k_range = k_values
    elbow_result = elbow_method(X, k_range)
    inertias = elbow_result['inertias']
    sil_scores = elbow_result['silhouette_scores']
    best_k = elbow_result['best_k']
    st.info(f"K đề xuất theo Silhouette Score: K={best_k} (điểm {max(sil_scores):.4f}).")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(selected_k, color='red', linestyle='--', label=f'K={selected_k}')
    ax1.set_xlabel('K'); ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    bar_colors = ['#FF6B6B' if ki == selected_k else '#4ECDC4' for ki in k_range]
    ax2.bar(list(k_range), sil_scores, color=bar_colors, edgecolor='white')
    ax2.set_xlabel('K'); ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Chạy KMeans ---
    with st.spinner("⏳ Đang chạy KMeans..."):
        km_result = kmeans_clustering(X, selected_k)
        labels = km_result['labels']

    df = analyze_clusters(df, labels)

    # Metrics
    n_clusters = len(set(labels))
    sil = km_result['silhouette']
    db_score = km_result['davies_bouldin']
    ch_score = km_result['calinski_harabasz']

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_clusters}</div><div class="metric-label">Số cụm</div></div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sil:.4f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)
    with mc3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{db_score:.4f}</div><div class="metric-label">Davies-Bouldin</div></div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{ch_score:.1f}</div><div class="metric-label">Calinski-Harabasz</div></div>', unsafe_allow_html=True)

    # --- Scatter Plot phân cụm ---
    st.markdown("#### 🗺️ Scatter Plot — Phân cụm theo Income & SpendingScore")
    fig, ax = plt.subplots(figsize=(10, 6))
    palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
    for i, cid in enumerate(sorted(df['Cluster'].unique())):
        subset = df[df['Cluster'] == cid]
        ax.scatter(subset['Income'], subset['SpendingScore'],
                   c=palette[i % len(palette)], label=f'Cụm {cid}',
                   s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Income (Thu nhập)', fontsize=12)
    ax.set_ylabel('SpendingScore (Điểm chi tiêu)', fontsize=12)
    ax.set_title('Kết quả phân cụm KMeans', fontweight='bold', fontsize=14)
    ax.legend(title='Cụm')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Scatter Plot: Age vs PurchaseFrequency ---
    st.markdown("#### 📊 Scatter Plot — Tuổi & Tần suất mua hàng")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, cid in enumerate(sorted(df['Cluster'].unique())):
        subset = df[df['Cluster'] == cid]
        ax.scatter(subset['Age'], subset['PurchaseFrequency'],
                   c=palette[i % len(palette)], label=f'Cụm {cid}',
                   s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel('Age (Tuổi)', fontsize=12)
    ax.set_ylabel('PurchaseFrequency (Tần suất)', fontsize=12)
    ax.set_title('Phân cụm theo Tuổi & Tần suất mua', fontweight='bold', fontsize=14)
    ax.legend(title='Cụm')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Mô tả nhóm khách hàng ---
    st.markdown("#### 🏷️ Mô tả nhóm khách hàng")
    for cid in sorted(df['Cluster'].unique()):
        grp = df[df['Cluster'] == cid]
        name = grp['ClusterName'].iloc[0]
        with st.expander(f"Cụm {cid} — {name} ({len(grp)} khách hàng)", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                st.metric("Tuổi TB", f"{grp['Age'].mean():.1f}" if 'Age' in grp.columns else "N/A")
            with cols[1]:
                st.metric("Thu nhập TB", f"{grp['Income'].mean():.1f}" if 'Income' in grp.columns else "N/A")
            with cols[2]:
                st.metric("Điểm chi tiêu TB", f"{grp['SpendingScore'].mean():.1f}" if 'SpendingScore' in grp.columns else "N/A")
            with cols[3]:
                st.metric("Tần suất mua TB", f"{grp['PurchaseFrequency'].mean():.1f}" if 'PurchaseFrequency' in grp.columns else "N/A")

    # --- Bảng dữ liệu ---
    st.markdown("#### 📋 Bảng dữ liệu phân cụm")
    display_cols = [c for c in ['CustomerID', 'Gender', 'Age', 'Income', 'SpendingScore', 'PurchaseFrequency', 'Cluster', 'ClusterName'] if c in df.columns]
    st.dataframe(df[display_cols], width="stretch", height=400)

    # Download
    csv = df[display_cols].to_csv(index=False, sep=';', encoding='utf-8-sig')
    st.download_button("⬇️ Tải kết quả CSV", csv, "ket_qua_phan_cum.csv", "text/csv", width="stretch")
