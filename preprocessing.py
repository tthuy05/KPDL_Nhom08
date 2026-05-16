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

warnings.filterwarnings('ignore')


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

    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]

    if len(missing_cols) == 0:
        print("   ✅ Không có missing values!")
        return df

    for col in missing_cols.index:
        count = missing_cols[col]
        if df[col].dtype in ['float64', 'int64']:
            # Cột số → điền median
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"   - {col}: {count} missing → điền median = {median_val}")
        else:
            # Cột chữ → điền mode
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"   - {col}: {count} missing → điền mode = '{mode_val}'")

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

    # Các cột dùng để clustering
    feature_cols = ['Age', 'Income', 'SpendingScore', 'PurchaseFrequency']
    feature_cols = [c for c in feature_cols if c in df.columns]

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
        'feature_cols': feature_cols
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
