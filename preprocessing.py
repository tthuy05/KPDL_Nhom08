# =============================================================================
# preprocessing.py - Tiền xử lý dữ liệu khách hàng
# =============================================================================
# Mô tả: Thực hiện các bước tiền xử lý đơn giản:
#         1. Xử lý missing values (điền median/mode)
#         2. Xóa dữ liệu trùng lặp
#         3. Mã hóa giới tính (Label Encoding)
#         4. Chuẩn hóa dữ liệu (StandardScaler)
# Tác giả: Nhóm 08 - Môn Khai Phá Dữ Liệu
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
import sys

warnings.filterwarnings('ignore')

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


FEATURE_COLUMNS = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']


def validate_required_columns(df, required_cols=None):
    """Kiem tra cac cot bat buoc truoc khi tien xu ly/phan cum."""
    required_cols = required_cols or FEATURE_COLUMNS
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Dataset thiếu cột bắt buộc: "
            + ", ".join(missing_cols)
            + ". Cần có các cột: "
            + ", ".join(required_cols)
        )


def validate_not_empty(df):
    """Dam bao dataset con du lieu sau khi lam sach."""
    if df.empty:
        raise ValueError("Dataset rỗng, không thể tiền xử lý hoặc phân cụm.")


def load_data(filepath):
    """
    Đọc dữ liệu từ file CSV.

    Parameters:
    -----------
    filepath : str
        Đường dẫn đến file CSV

    Returns:
    --------
    pd.DataFrame
    """
    print(f"📂 Đang đọc dữ liệu từ: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   - Kích thước: {df.shape[0]} dòng x {df.shape[1]} cột")
    return df


def handle_missing_values(df):
    """
    Xử lý giá trị bị thiếu (missing values).

    Chiến lược:
    - Cột số (numeric): điền bằng trung vị (median)
    - Cột chữ (categorical): điền bằng giá trị phổ biến nhất (mode)
    """
    print("\n🔧 Xử lý Missing Values:")
    print("-" * 40)

    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]

    if len(missing_cols) == 0:
        print("   ✅ Không có missing values!")
        return df

    for col in missing_cols.index:
        count = missing_cols[col]
        if pd.api.types.is_numeric_dtype(df[col]):
            # Cột số → điền median
            median_val = df[col].median()
            if pd.isna(median_val):
                raise ValueError(
                    f"Cột {col} toàn missing, không thể điền bằng median. "
                    "Hãy bổ sung dữ liệu hợp lệ hoặc bỏ cột này khỏi feature."
                )
            df[col] = df[col].fillna(median_val)
            print(f"   - {col}: {count} missing → điền median = {median_val}")
        else:
            # Cột chữ → điền mode
            mode_series = df[col].mode(dropna=True)
            mode_val = mode_series.iloc[0] if not mode_series.empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f"   - {col}: {count} missing → điền mode = '{mode_val}'")

    missing_after = df.isnull().sum().sum()
    if missing_after > 0:
        raise ValueError(f"Vẫn còn {missing_after} missing values sau tiền xử lý.")

    print(f"   ✅ Đã xử lý xong {missing_before.sum()} missing values!")
    return df


def remove_duplicates(df):
    """Xóa các dòng dữ liệu trùng lặp."""
    print("\n🔧 Xử lý Duplicates:")
    print("-" * 40)

    n_dup = df.duplicated().sum()
    if n_dup == 0:
        print("   ✅ Không có dữ liệu trùng lặp!")
        return df

    df = df.drop_duplicates().reset_index(drop=True)
    validate_not_empty(df)
    print(f"   - Đã xóa {n_dup} dòng trùng lặp")
    print(f"   - Kích thước mới: {df.shape[0]} dòng")
    return df


def encode_gender(df):
    """
    Mã hóa cột Gender bằng Label Encoding.
    Ví dụ: 'Nam' → 0, 'Nữ' → 1
    """
    print("\n🔧 Label Encoding (Gender):")
    print("-" * 40)

    le = LabelEncoder()
    if 'Gender' in df.columns:
        df = df.copy()
        df['Gender'] = df['Gender'].astype(str)
        df['Gender_Encoded'] = le.fit_transform(df['Gender'])
        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"   - Gender: {mapping}")
        print("   ✅ Mã hóa thành công!")
    return df, le


def scale_features(df):
    """
    Chuẩn hóa dữ liệu bằng StandardScaler.

    Công thức: z = (x - mean) / std
    → Đưa tất cả features về cùng thang đo (mean=0, std=1)
    → Giúp KMeans hoạt động tốt hơn vì KMeans dựa trên khoảng cách
    """
    print("\n🔧 Chuẩn hóa dữ liệu (StandardScaler):")
    print("-" * 40)

    validate_required_columns(df, FEATURE_COLUMNS)
    validate_not_empty(df)

    feature_cols = FEATURE_COLUMNS.copy()
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if df[feature_cols].isnull().sum().sum() > 0:
        missing_detail = df[feature_cols].isnull().sum()
        missing_detail = missing_detail[missing_detail > 0]
        raise ValueError(
            "Các feature dùng để phân cụm còn missing hoặc không phải số: "
            + missing_detail.to_dict().__repr__()
        )

    print(f"   - Features: {feature_cols}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])

    print(f"   - Shape: {X_scaled.shape}")
    print("   ✅ Chuẩn hóa thành công!")

    return X_scaled, scaler, feature_cols


def preprocess_pipeline(df):
    """
    Pipeline tiền xử lý hoàn chỉnh.

    Thực hiện tuần tự:
    1. Xử lý missing values
    2. Xóa duplicates
    3. Mã hóa Gender
    4. Chuẩn hóa dữ liệu

    Returns:
    --------
    dict chứa: df_cleaned, X_scaled, scaler, feature_cols
    """
    print("=" * 50)
    print("🚀 BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 50)

    df = df.copy()
    validate_required_columns(df, FEATURE_COLUMNS)
    validate_not_empty(df)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df, encoder = encode_gender(df)
    X_scaled, scaler, feature_cols = scale_features(df)

    print("\n" + "=" * 50)
    print("✅ TIỀN XỬ LÝ HOÀN TẤT!")
    print(f"   - Kích thước cuối: {df.shape}")
    print("=" * 50)

    return {
        'df_cleaned': df,
        'X_scaled': X_scaled,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'encoder': encoder
    }


# =============================================================================
# MAIN - Chạy trực tiếp để test tiền xử lý
# =============================================================================
if __name__ == '__main__':
    import os

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'customers.csv')

    if not os.path.exists(data_path):
        print("⚠️ Chưa có file customers.csv!")
        print("   Hãy chạy: python generate_data.py")
    else:
        df = load_data(data_path)
        result = preprocess_pipeline(df)
        print("\n📊 Dữ liệu sau tiền xử lý:")
        print(result['df_cleaned'].head())
