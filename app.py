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
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import os

from clustering import analyze_clusters, elbow_method, get_valid_k_range, kmeans_clustering
from preprocessing import preprocess_pipeline

warnings.filterwarnings('ignore')

# ======================== CHART STYLE ========================
TEAL    = '#0D9488'
TEAL_D  = '#0F766E'
AMBER   = '#D97706'
SLATE   = '#1E293B'

CLUSTER_PALETTE = ['#0D9488', '#7C3AED', '#E11D48', '#D97706',
                   '#2563EB', '#059669', '#DB2777', '#CA8A04',
                   '#6366F1', '#65A30D']
HIST_PALETTE    = ['#0D9488', '#7C3AED', '#E11D48', '#D97706']

plt.rcParams.update({
    'figure.facecolor': '#FFFFFF',
    'figure.dpi':       100,
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#CBD5E1',
    'axes.labelcolor':  '#334155',
    'axes.titlesize':   12,
    'axes.titleweight': 600,
    'axes.labelsize':   10,
    'axes.grid':        True,
    'grid.color':       '#E2E8F0',
    'grid.alpha':       0.5,
    'grid.linewidth':   0.4,
    'xtick.color':      '#64748B',
    'ytick.color':      '#64748B',
    'xtick.labelsize':  8.5,
    'ytick.labelsize':  8.5,
    'font.family':      'sans-serif',
    'font.sans-serif':  ['Segoe UI', 'Helvetica Neue', 'Arial'],
    'legend.framealpha': 0,
    'legend.fontsize':   8.5,
})

# ======================== CẤU HÌNH TRANG ========================
st.set_page_config(
    page_title="Phân Cụm Khách Hàng — Nhóm 08",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================== CUSTOM CSS ========================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

    /* -------- global -------- */
    html, body, .stApp {
        font-family: 'DM Sans', 'Segoe UI', system-ui, sans-serif !important;
    }
    .block-container { padding-top: 2rem; }

    /* -------- header banner -------- */
    .hero {
        background: #0F766E;
        background-image: linear-gradient(135deg, #0F766E 0%, #0D9488 60%, #14B8A6 100%);
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute; right: -40px; top: -40px;
        width: 180px; height: 180px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .hero::after {
        content: '';
        position: absolute; right: 80px; bottom: -30px;
        width: 100px; height: 100px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .hero h1 {
        color: #FFFFFF; margin: 0;
        font-size: 1.5rem; font-weight: 700;
        letter-spacing: -0.01em;
    }
    .hero p {
        color: rgba(255,255,255,0.78);
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem; font-weight: 400;
    }

    /* -------- KPI cards -------- */
    .kpi-row { display: flex; gap: 1rem; margin: 0.5rem 0 1.2rem 0; }
    .kpi {
        flex: 1;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.1rem 1.2rem;
        text-align: center;
        transition: transform 0.12s, box-shadow 0.12s;
    }
    .kpi:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .kpi .val {
        font-size: 1.6rem; font-weight: 700;
        color: #0F766E;
        line-height: 1.2;
    }
    .kpi .lbl {
        font-size: 0.72rem; font-weight: 500;
        color: #64748B;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .kpi.neutral .val { color: #334155; }

    /* -------- section heading -------- */
    .sec {
        font-size: 0.82rem; font-weight: 600;
        color: #0F766E;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 2rem 0 0.6rem 0;
    }

    /* -------- sidebar -------- */
    section[data-testid="stSidebar"] {
        background: #F0FDFA !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #134E4A !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* -------- buttons -------- */
    .stButton > button[kind="primary"],
    .stDownloadButton > button {
        background: #0D9488 !important;
        color: #FFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: background 0.15s;
    }
    .stButton > button[kind="primary"]:hover,
    .stDownloadButton > button:hover {
        background: #0F766E !important;
    }

    /* -------- chart wrapper -------- */
    .chart-label {
        font-size: 0.88rem; font-weight: 600;
        color: #334155; margin: 1rem 0 0.4rem 0;
    }

    /* -------- footer -------- */
    .foot {
        text-align: center; font-size: 0.72rem;
        color: #94A3B8; padding: 1.5rem 0 0.5rem;
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


def kpi_row(items):
    """Render a horizontal row of KPI cards.
    items: list of (value, label, accent:bool)
    """
    cards = ''
    for val, lbl, accent in items:
        cls = 'kpi' if accent else 'kpi neutral'
        cards += f'<div class="{cls}"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
    return f'<div class="kpi-row">{cards}</div>'


# ======================== SIDEBAR ========================
with st.sidebar:
    st.markdown("### Cấu hình")

    st.markdown("**Nguồn dữ liệu**")
    upload = st.file_uploader("Upload CSV", type=['csv'])
    use_default = st.checkbox("Dùng dataset mặc định", value=True)

    st.divider()
    st.markdown("**KMeans**")
    k = st.slider("Số cụm (K)", 2, 10, 8)

    st.divider()
    st.markdown("**Biểu đồ EDA**")
    show_hist = st.checkbox("Histogram", value=True)
    show_heatmap = st.checkbox("Heatmap tương quan", value=True)

    run_btn = st.button("Chạy phân cụm", type="primary", use_container_width=True)

    st.divider()
    st.markdown('<div class="foot">Nhóm 08 · Khai Phá Dữ Liệu · 2026</div>', unsafe_allow_html=True)


# ======================== NỘI DUNG CHÍNH ========================
st.markdown("""
<div class="hero">
    <h1>Phân cụm khách hàng</h1>
    <p>Đồ án Khai Phá Dữ Liệu · Nhóm 08 · KMeans Clustering</p>
</div>
""", unsafe_allow_html=True)

# Load data
df_raw = None
if upload is not None:
    df_raw = pd.read_csv(upload)
    st.success(f"Đã tải **{upload.name}** — {len(df_raw):,} dòng")
elif use_default:
    df_raw = load_default_data()
    if df_raw is not None:
        st.info(f"Đang dùng dataset mặc định — {len(df_raw):,} dòng")
    else:
        st.warning("Chưa có dataset mặc định. Đặt file `customers_kaggle.csv` vào thư mục `data/` hoặc upload CSV.")

if df_raw is None:
    st.stop()

# Tiền xử lý
try:
    preprocess_result = preprocess_pipeline(df_raw)
except ValueError as exc:
    st.error(f"Không thể tiền xử lý dữ liệu: {exc}")
    st.info("Upload file có các cột: CustomerID, Gender, Age, Income, SpendingScore, PurchaseFrequency.")
    st.stop()

df = preprocess_result['df_cleaned']
df_input = preprocess_result['df_input']
X_scaled = preprocess_result['X_scaled']
feature_cols = preprocess_result['feature_cols']

# ---- TỔNG QUAN DỮ LIỆU ----
st.markdown('<div class="sec">Tổng quan dữ liệu</div>', unsafe_allow_html=True)

missing = df_input.isnull().sum().sum()
dup = df_input.duplicated().sum()
st.markdown(kpi_row([
    (f"{len(df):,}", "Khách hàng", True),
    (str(missing), "Missing values (raw)", False),
    (str(dup), "Duplicates (raw)", False),
]), unsafe_allow_html=True)

with st.expander("Xem dữ liệu mẫu"):
    st.dataframe(df.head(15), width="stretch")

# ---- EDA ----
st.markdown('<div class="sec">Khám phá dữ liệu</div>', unsafe_allow_html=True)

num_cols = feature_cols

if show_hist:
    st.markdown('<div class="chart-label">Phân phối dữ liệu</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, len(num_cols), figsize=(15, 3.5))
    if len(num_cols) == 1:
        axes = [axes]
    for i, col in enumerate(num_cols):
        color = HIST_PALETTE[i % len(HIST_PALETTE)]
        axes[i].hist(df[col].dropna(), bins=30, color=color,
                     edgecolor='#FFFFFF', alpha=0.85, linewidth=0.3)
        axes[i].set_title(col, fontsize=10.5, fontweight=600, color='#334155')
        m = df[col].mean()
        axes[i].axvline(m, color='#DC2626', linestyle='--', linewidth=0.9,
                        label=f'TB: {m:.1f}')
        axes[i].legend(fontsize=7, loc='upper right', framealpha=0)
        axes[i].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

if show_heatmap:
    st.markdown('<div class="chart-label">Ma trận tương quan</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(160, 10, s=80, l=55, as_cmap=True)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=0.6, ax=ax,
                vmin=-1, vmax=1,
                annot_kws={'size': 9.5, 'weight': 'bold'},
                cbar_kws={'shrink': 0.72})
    ax.set_title('Hệ số tương quan Pearson', fontsize=11, fontweight=600,
                 color='#334155', pad=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ======================== PHÂN CỤM ========================
if run_btn:
    st.markdown('<div class="sec">Kết quả phân cụm</div>', unsafe_allow_html=True)

    X = X_scaled

    try:
        k_values = list(get_valid_k_range(X))
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    selected_k = min(k, max(k_values))
    if selected_k != k:
        st.warning(f"K={k} không hợp lệ. Đã dùng K={selected_k}.")

    # Elbow
    st.markdown('<div class="chart-label">Elbow Method & Silhouette Score</div>', unsafe_allow_html=True)
    k_range = k_values
    elbow_result = elbow_method(X, k_range)
    inertias = elbow_result['inertias']
    sil_scores = elbow_result['silhouette_scores']
    best_k = elbow_result['best_k']
    st.info(f"K đề xuất theo Silhouette: **K = {best_k}** (score = {max(sil_scores):.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.2))
    # elbow line
    ax1.plot(list(k_range), inertias, color=TEAL, linewidth=2,
             marker='o', markersize=6, markerfacecolor='#FFF',
             markeredgewidth=1.8, markeredgecolor=TEAL)
    ax1.axvline(selected_k, color='#DC2626', linestyle='--', linewidth=0.9,
                label=f'K = {selected_k}')
    ax1.set_xlabel('K')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.legend(framealpha=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # silhouette bars
    bar_c = [TEAL if ki == selected_k else '#99F6E4' for ki in k_range]
    ax2.bar(list(k_range), sil_scores, color=bar_c, width=0.6,
            edgecolor='#FFFFFF', linewidth=0.3)
    ax2.set_xlabel('K')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # KMeans
    with st.spinner("Đang phân cụm..."):
        km_result = kmeans_clustering(X, selected_k)
        labels = km_result['labels']

    df = analyze_clusters(df, labels)

    n_clusters = len(set(labels))
    sil = km_result['silhouette']
    db_score = km_result['davies_bouldin']
    ch_score = km_result['calinski_harabasz']

    st.markdown(kpi_row([
        (str(n_clusters), "Số cụm", True),
        (f"{sil:.4f}", "Silhouette", False),
        (f"{db_score:.4f}", "Davies-Bouldin", False),
        (f"{ch_score:.1f}", "Calinski-Harabasz", False),
    ]), unsafe_allow_html=True)

    # ---- Scatter plots ----
    def scatter(df, x, y, xlabel, ylabel, title):
        fig, ax = plt.subplots(figsize=(10, 5.2))
        for i, cid in enumerate(sorted(df['Cluster'].unique())):
            s = df[df['Cluster'] == cid]
            ax.scatter(s[x], s[y],
                       c=CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)],
                       label=f'Cụm {cid}', s=36, alpha=0.72,
                       edgecolors='#FFF', linewidth=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, fontweight=600, color='#334155')
        ax.legend(title='Cụm', title_fontsize=8.5, loc='best', framealpha=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        return fig

    st.markdown('<div class="chart-label">Thu nhập vs Chi tiêu</div>', unsafe_allow_html=True)
    fig = scatter(df, 'Income', 'SpendingScore',
                  'Income (Thu nhập)', 'Spending Score', 'Phân cụm theo Thu nhập & Chi tiêu')
    st.pyplot(fig); plt.close()

    st.markdown('<div class="chart-label">Tuổi vs Tần suất mua</div>', unsafe_allow_html=True)
    fig = scatter(df, 'Age', 'PurchaseFrequency',
                  'Age (Tuổi)', 'Purchase Frequency', 'Phân cụm theo Tuổi & Tần suất mua')
    st.pyplot(fig); plt.close()

    # ---- Mô tả cụm ----
    st.markdown('<div class="chart-label">Đặc điểm từng cụm</div>', unsafe_allow_html=True)
    for cid in sorted(df['Cluster'].unique()):
        grp = df[df['Cluster'] == cid]
        name = grp['ClusterName'].iloc[0]
        with st.expander(f"Cụm {cid} — {name}  ({len(grp)} khách)", expanded=True):
            cols = st.columns(4)
            with cols[0]:
                st.metric("Tuổi TB", f"{grp['Age'].mean():.1f}" if 'Age' in grp.columns else "—")
            with cols[1]:
                st.metric("Thu nhập TB", f"{grp['Income'].mean():.1f}" if 'Income' in grp.columns else "—")
            with cols[2]:
                st.metric("Chi tiêu TB", f"{grp['SpendingScore'].mean():.1f}" if 'SpendingScore' in grp.columns else "—")
            with cols[3]:
                st.metric("Tần suất mua TB", f"{grp['PurchaseFrequency'].mean():.1f}" if 'PurchaseFrequency' in grp.columns else "—")

    # ---- Bảng dữ liệu ----
    st.markdown('<div class="chart-label">Bảng dữ liệu</div>', unsafe_allow_html=True)
    display_cols = [c for c in ['CustomerID', 'Gender', 'Age', 'Income', 'SpendingScore',
                                'PurchaseFrequency', 'Cluster', 'ClusterName'] if c in df.columns]
    st.dataframe(df[display_cols], width="stretch", height=400)

    csv = df[display_cols].to_csv(index=False, sep=';', encoding='utf-8-sig')
    st.download_button("Tải kết quả CSV", csv, "ket_qua_phan_cum.csv", "text/csv",
                       use_container_width=True)
