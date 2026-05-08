# =============================================================================
# app.py - Dashboard Streamlit
# =============================================================================
# Mô tả: Giao diện web tương tác cho phân cụm khách hàng
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')

# ======================== CẤU HÌNH TRANG ========================
st.set_page_config(
    page_title="Ứng dụng kỹ thuật gom nhóm trong khai thác dữ liệu khách hàng để phân khúc thị trường",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        margin: 0; font-size: 2rem; font-weight: 700;
    }
    .main-header p {
        margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 1.05rem;
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 14px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-value {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-label { font-size: 0.9rem; color: #666; margin-top: 0.3rem; }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #667eea22, transparent);
        padding: 0.8rem 1.2rem;
        border-left: 4px solid #667eea;
        border-radius: 0 8px 8px 0;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600; font-size: 1.15rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #e8eaf6 0%, #f3e5f5 50%, #ede7f6 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown { color: #333; }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #4a148c !important; }

    /* Tables */
    .dataframe { border-radius: 8px; overflow: hidden; }

    /* Cluster badge */
    .cluster-badge {
        display: inline-block; padding: 0.3rem 1rem;
        border-radius: 20px; font-weight: 600; font-size: 0.85rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ======================== HÀM TIỆN ÍCH ========================
@st.cache_data
def load_default_data():
    """Tải dataset mặc định từ thư mục data/"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def preprocess(df):
    """Pipeline tiền xử lý nhanh cho dashboard"""
    df = df.copy()
    # Missing values → median / mode
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    # Duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    # Label Encoding
    cat_cols = ['Gender', 'MembershipLevel', 'PreferredCategory']
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_Encoded'] = le.fit_transform(df[col])
            encoders[col] = le
    return df, encoders


def get_features(df):
    """Lấy danh sách cột features cho clustering"""
    candidates = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency',
                  'AverageOrderValue', 'LastPurchaseDays',
                  'Gender_Encoded', 'MembershipLevel_Encoded', 'PreferredCategory_Encoded']
    return [c for c in candidates if c in df.columns]


def run_clustering(X, algo, params):
    """Chạy thuật toán clustering được chọn"""
    if algo == 'KMeans':
        model = KMeans(n_clusters=params['k'], random_state=42, n_init=10)
    elif algo == 'DBSCAN':
        model = DBSCAN(eps=params['eps'], min_samples=params['min_samples'])
    else:
        model = AgglomerativeClustering(n_clusters=params['k'], linkage=params.get('linkage', 'ward'))
    return model.fit_predict(X), model


def assign_cluster_names(df):
    """Gán tên mô tả cho từng cụm dựa trên đặc điểm"""
    df = df.copy()
    cluster_names = {}
    valid = df[df['Cluster'] != -1]
    if valid.empty:
        df['ClusterName'] = 'Noise'
        return df

    inc_med = valid['Income'].median() if 'Income' in valid.columns else 0
    spe_med = valid['SpendingScore'].median() if 'SpendingScore' in valid.columns else 0

    for cid in sorted(valid['Cluster'].unique()):
        s = valid[valid['Cluster'] == cid]
        inc = s['Income'].mean() if 'Income' in s.columns else 0
        spe = s['SpendingScore'].mean() if 'SpendingScore' in s.columns else 0
        freq = s['PurchaseFrequency'].mean() if 'PurchaseFrequency' in s.columns else 0
        aov = s['AverageOrderValue'].mean() if 'AverageOrderValue' in s.columns else 0
        lpd = s['LastPurchaseDays'].mean() if 'LastPurchaseDays' in s.columns else 0

        if inc > inc_med * 1.3 and spe > spe_med * 1.2:
            cluster_names[cid] = "💎 Khách VIP"
        elif freq > valid['PurchaseFrequency'].median() * 1.3 and aov < valid['AverageOrderValue'].median() * 0.8:
            cluster_names[cid] = "🏷️ Khách săn giảm giá"
        elif inc > inc_med * 0.8 and spe > spe_med * 0.8:
            cluster_names[cid] = "⭐ Khách tiềm năng"
        elif lpd > valid['LastPurchaseDays'].median() * 1.5:
            cluster_names[cid] = "😴 Khách không hoạt động"
        else:
            cluster_names[cid] = "💰 Khách tiết kiệm"

    df['ClusterName'] = df['Cluster'].map(cluster_names).fillna("🔇 Noise")
    return df


# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("## 🎛️ Bảng Điều Khiển")
    st.markdown("---")

    # Upload hoặc dùng mặc định
    st.markdown("### 📁 Dữ liệu")
    upload = st.file_uploader("Upload file CSV", type=['csv'])
    use_default = st.checkbox("Dùng dataset mặc định", value=True)

    st.markdown("---")
    st.markdown("### ⚙️ Thuật toán")
    algorithm = st.selectbox("Chọn thuật toán", ['KMeans', 'DBSCAN', 'Agglomerative'])

    params = {}
    if algorithm in ['KMeans', 'Agglomerative']:
        params['k'] = st.slider("Số cụm (K)", 2, 10, 4)
        if algorithm == 'Agglomerative':
            params['linkage'] = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])
    else:
        params['eps'] = st.slider("Epsilon (eps)", 0.5, 5.0, 1.5, 0.1)
        params['min_samples'] = st.slider("Min Samples", 2, 20, 5)

    st.markdown("---")
    st.markdown("### 📊 Biểu đồ")
    show_hist = st.checkbox("Histogram", value=True)
    show_heatmap = st.checkbox("Heatmap", value=True)
    show_box = st.checkbox("Boxplot", value=True)
    show_pair = st.checkbox("Pairplot", value=False)

    run_btn = st.button("🚀 Chạy Phân Cụm", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; opacity:0.7; font-size:0.8rem; color:#aaa;'>
        Nhóm 08 — KPDL<br>
        © 2026
    </div>
    """, unsafe_allow_html=True)


# ======================== NỘI DUNG CHÍNH ========================
# Header
st.markdown("""
<div class="main-header">
    <h1>📊 Phân Cụm Khách Hàng</h1>
    <p>Ứng dụng kỹ thuật gom nhóm trong khai thác dữ liệu khách hàng để phân khúc thị trường</p>
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
        st.info(f"📂 Đang dùng dataset mặc định ({len(df_raw)} dòng)")
    else:
        st.warning("⚠️ Chưa có dataset mặc định. Hãy chạy `python generate_data.py` trước hoặc upload file CSV.")

if df_raw is None:
    st.stop()

# Tiền xử lý
df, encoders = preprocess(df_raw)

# ---- METRIC CARDS ----
st.markdown('<div class="section-header">📋 Tổng Quan Dữ Liệu</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Khách hàng</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df.columns)}</div><div class="metric-label">Đặc trưng</div></div>', unsafe_allow_html=True)
with c3:
    missing = df_raw.isnull().sum().sum()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{missing}</div><div class="metric-label">Missing (gốc)</div></div>', unsafe_allow_html=True)
with c4:
    dup = df_raw.duplicated().sum()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{dup}</div><div class="metric-label">Duplicates (gốc)</div></div>', unsafe_allow_html=True)

# ---- DỮ LIỆU MẪU ----
with st.expander("🔍 Xem dữ liệu mẫu", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)

# ---- BIỂU ĐỒ KHÁM PHÁ ----
st.markdown('<div class="section-header">📈 Khám Phá Dữ Liệu (EDA)</div>', unsafe_allow_html=True)

num_cols = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays']
num_cols = [c for c in num_cols if c in df.columns]

if show_hist:
    st.markdown("#### 📊 Histogram — Phân phối dữ liệu")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i, col in enumerate(num_cols[:6]):
        axes[i].hist(df[col].dropna(), bins=30, color=colors[i], edgecolor='white', alpha=0.85)
        axes[i].set_title(col, fontweight='bold')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1)
    for j in range(len(num_cols), 6):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

if show_heatmap:
    st.markdown("#### 🌡️ Heatmap — Ma trận tương quan")
    fig, ax = plt.subplots(figsize=(10, 7))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, square=True, linewidths=1, ax=ax)
    ax.set_title('Ma tran tuong quan', fontweight='bold', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

if show_box:
    st.markdown("#### 📦 Boxplot — Phát hiện Outliers")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    for i, col in enumerate(num_cols[:6]):
        bp = axes[i].boxplot(df[col].dropna(), patch_artist=True, vert=True)
        bp['boxes'][0].set_facecolor(colors[i])
        bp['boxes'][0].set_alpha(0.7)
        axes[i].set_title(col, fontweight='bold')
    for j in range(len(num_cols), 6):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

if show_pair:
    st.markdown("#### 🔗 Pairplot — Quan hệ giữa các biến")
    pair_cols = [c for c in ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency'] if c in df.columns]
    g = sns.pairplot(df[pair_cols], diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20}, height=2.2)
    st.pyplot(g.figure)
    plt.close()

# ======================== CLUSTERING ========================
if run_btn:
    st.markdown('<div class="section-header">🤖 Kết Quả Phân Cụm</div>', unsafe_allow_html=True)

    features = get_features(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])

    # --- Elbow Method (chỉ cho KMeans) ---
    if algorithm == 'KMeans':
        st.markdown("#### 📐 Elbow Method & Silhouette Score")
        k_range = range(2, 11)
        inertias, sil_scores = [], []
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X, km.labels_))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(params['k'], color='red', linestyle='--', label=f"K={params['k']}")
        ax1.set_xlabel('K'); ax1.set_ylabel('Inertia'); ax1.set_title('Elbow Method', fontweight='bold')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        bar_colors = ['#FF6B6B' if k == params['k'] else '#4ECDC4' for k in k_range]
        ax2.bar(list(k_range), sil_scores, color=bar_colors, edgecolor='white')
        ax2.set_xlabel('K'); ax2.set_ylabel('Silhouette'); ax2.set_title('Silhouette Score', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # --- Chạy clustering ---
    with st.spinner(f"⏳ Đang chạy {algorithm}..."):
        labels, model = run_clustering(X, algorithm, params)

    df['Cluster'] = labels
    df = assign_cluster_names(df)

    # Metrics
    valid_mask = labels != -1
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_clusters}</div><div class="metric-label">Số cụm</div></div>', unsafe_allow_html=True)
    with mc2:
        sil = silhouette_score(X[valid_mask], labels[valid_mask]) if valid_mask.sum() > n_clusters and n_clusters >= 2 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{sil:.4f}</div><div class="metric-label">Silhouette Score</div></div>', unsafe_allow_html=True)
    with mc3:
        ch = calinski_harabasz_score(X[valid_mask], labels[valid_mask]) if valid_mask.sum() > n_clusters and n_clusters >= 2 else 0
        st.markdown(f'<div class="metric-card"><div class="metric-value">{ch:.1f}</div><div class="metric-label">Calinski-Harabasz</div></div>', unsafe_allow_html=True)
    with mc4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{n_noise}</div><div class="metric-label">Noise Points</div></div>', unsafe_allow_html=True)

    # --- PCA 2D Scatter ---
    st.markdown("#### 🗺️ Trực quan hóa cụm (PCA 2D)")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 7))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.viridis(np.linspace(0, 1, max(len(unique_labels), 1)))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        name = 'Noise' if lab == -1 else f'Cụm {lab}'
        color = 'gray' if lab == -1 else cmap[i]
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=name,
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax.set_title(f'Kết quả {algorithm} trên PCA 2D', fontweight='bold', fontsize=14)
    ax.legend(title='Cụm', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- So sánh đặc trưng ---
    st.markdown("#### 📊 So sánh đặc trưng giữa các cụm")
    valid_df = df[df['Cluster'] != -1]
    compare_cols = [c for c in num_cols if c in valid_df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    palette = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    cluster_means = valid_df.groupby('Cluster')[compare_cols].mean()

    for i, col in enumerate(compare_cols[:6]):
        bars = axes[i].bar(cluster_means.index.astype(str), cluster_means[col],
                           color=palette, edgecolor='white', alpha=0.85)
        axes[i].set_title(col, fontweight='bold')
        axes[i].set_xlabel('Cụm')
        for bar, val in zip(bars, cluster_means[col]):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                         f'{val:.1f}', ha='center', va='bottom', fontsize=8)
    for j in range(len(compare_cols), 6):
        axes[j].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # --- Phân tích nhóm ---
    st.markdown("#### 🏷️ Phân loại nhóm khách hàng")
    for cid in sorted(valid_df['Cluster'].unique()):
        grp = valid_df[valid_df['Cluster'] == cid]
        name = grp['ClusterName'].iloc[0]
        with st.expander(f"Cụm {cid} — {name} ({len(grp)} khách hàng)", expanded=True):
            cols = st.columns(3)
            with cols[0]:
                st.metric("Thu nhập TB", f"{grp['Income'].mean():.1f}" if 'Income' in grp.columns else "N/A")
                st.metric("Tuổi TB", f"{grp['Age'].mean():.1f}" if 'Age' in grp.columns else "N/A")
            with cols[1]:
                st.metric("Điểm chi tiêu TB", f"{grp['SpendingScore'].mean():.1f}" if 'SpendingScore' in grp.columns else "N/A")
                st.metric("Tần suất mua TB", f"{grp['PurchaseFrequency'].mean():.1f}" if 'PurchaseFrequency' in grp.columns else "N/A")
            with cols[2]:
                st.metric("Giá trị đơn TB", f"{grp['AverageOrderValue'].mean():.2f}" if 'AverageOrderValue' in grp.columns else "N/A")
                st.metric("Ngày mua cuối TB", f"{grp['LastPurchaseDays'].mean():.0f}" if 'LastPurchaseDays' in grp.columns else "N/A")

    # --- Bảng dữ liệu ---
    st.markdown("#### 📋 Bảng dữ liệu phân cụm")
    display_cols = ['CustomerID', 'Gender', 'Age', 'Income', 'SpendingScore',
                    'PurchaseFrequency', 'AverageOrderValue', 'LastPurchaseDays',
                    'MembershipLevel', 'PreferredCategory', 'Cluster', 'ClusterName']
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, height=400)

    # Download
    csv = df[display_cols].to_csv(index=False, encoding='utf-8-sig')
    st.download_button("⬇️ Tải kết quả CSV", csv, "clustered_customers.csv", "text/csv",
                       use_container_width=True)
