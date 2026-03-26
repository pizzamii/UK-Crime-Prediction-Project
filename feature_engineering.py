"""
特征工程模块
============
在 data_preprocessing.py 输出的 processed_crime_data.csv 基础上，构建时空特征：

  空间特征：
    - 经纬度网格化编码（geo_grid，grid_size=0.1°≈11km）
    - 到英国主要城市中心的欧式距离（5个城市）

  犯罪密度特征（SHAP 识别出最重要的特征组）：
    - 按 geo_grid 统计各犯罪类型的空间密度（含 bicycle theft、anti-social behaviour）
    - total_crimes：目标变量（每个网格的总犯罪数）

  时间聚合特征：
    - 按 (geo_grid, month) 的月度犯罪数统计量
      (monthly_crime_mean, monthly_crime_std, monthly_crime_max, monthly_crime_range)

  分类特征编码：
    - 低基数（< 15 类）→ 独热编码
    - 高基数（≥ 15 类）→ 标签编码

输入:  processed_crime_data.csv
输出:  featured_crime_data.csv  +  X_features.npy  +  feature_names.pkl  +  scaler.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
import argparse

# StandardScaler 可选依赖（model_training 阶段在用户本机运行时会用到）
try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── 数据加载 ─────────────────────────────────────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    print(f"加载数据: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}，请先运行 data_preprocessing.py")
    df = pd.read_csv(file_path)
    print(f"  {df.shape[0]:,} 行  x  {df.shape[1]} 列")
    return df


# ── 空间特征 ──────────────────────────────────────────────────────────────────

def add_geocoding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    地理编码特征：
      - lon_grid / lat_grid：将经纬度量化为整数格（grid_size=0.1°）
      - geo_grid：字符串格网 ID（用于聚合）
      - dist_to_{city}：到5个主要城市的欧式距离（度为单位，用于相对比较）
    """
    print("添加地理编码特征...")

    grid_size = 0.1
    df['lon_grid'] = (df['longitude'] / grid_size).astype(int)
    df['lat_grid'] = (df['latitude']  / grid_size).astype(int)
    df['geo_grid'] = df['lon_grid'].astype(str) + '_' + df['lat_grid'].astype(str)

    # 5 个主要英国城市中心坐标（经度, 纬度）
    CITY_CENTRES = {
        'london':     (-0.1278, 51.5074),
        'manchester': (-2.2426, 53.4808),
        'birmingham': (-1.8904, 52.4862),
        'leeds':      (-1.5491, 53.8008),
        'glasgow':    (-4.2518, 55.8642),
    }
    for city, (lon, lat) in CITY_CENTRES.items():
        df[f'dist_to_{city}'] = np.sqrt(
            (df['longitude'] - lon) ** 2 + (df['latitude'] - lat) ** 2
        )

    print(f"  网格数量: {df['geo_grid'].nunique():,}")
    return df


# ── 犯罪密度特征（目标变量 total_crimes 所在模块）────────────────────────────

def add_crime_density_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 geo_grid 统计各犯罪类型的空间密度：
      - {crime_type} 列：该网格内该类型犯罪的总数（pivot 展开）
      - total_crimes：该网格的犯罪总数（即模型目标变量）

    注：此处的密度是全数据集上的，不会泄露月度信息。
    """
    print("添加犯罪密度特征（目标变量 total_crimes）...")

    if 'crime_type' not in df.columns:
        print("  警告: 缺少 crime_type 列，跳过")
        return df

    # 按 (geo_grid, crime_type) 聚合
    counts = df.groupby(['geo_grid', 'crime_type']).size().reset_index(name='cnt')
    pivot  = counts.pivot(index='geo_grid', columns='crime_type', values='cnt').fillna(0)
    pivot['total_crimes'] = pivot.sum(axis=1)

    # 列名标准化（去掉特殊字符，方便后续引用）
    pivot.columns = [str(c).replace(' ', '_').replace('-', '_').replace('/', '_')
                     for c in pivot.columns]
    pivot = pivot.reset_index()

    df = pd.merge(df, pivot, on='geo_grid', how='left')
    print(f"  添加了 {len(pivot.columns) - 1} 个犯罪密度列（含 total_crimes）")
    return df


# ── 时间聚合特征 ──────────────────────────────────────────────────────────────

def add_temporal_aggregated_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    按 (geo_grid, month) 统计月度犯罪数，然后对 geo_grid 做统计汇总：
      - monthly_crime_mean：该网格的月均犯罪数
      - monthly_crime_std：月度标准差
      - monthly_crime_max：月度峰值
      - monthly_crime_range：峰谷差（max - min）
    """
    print("添加时间聚合特征...")

    if 'geo_grid' not in df.columns or 'month' not in df.columns:
        print("  缺少必要列（geo_grid / month），跳过")
        return df

    monthly = df.groupby(['geo_grid', 'month']).size().reset_index(name='monthly_cnt')
    stats   = monthly.groupby('geo_grid')['monthly_cnt'].agg(
        monthly_crime_mean='mean',
        monthly_crime_std='std',
        monthly_crime_max='max',
        monthly_crime_min='min',
    ).reset_index()
    stats['monthly_crime_std']   = stats['monthly_crime_std'].fillna(0)
    stats['monthly_crime_range'] = stats['monthly_crime_max'] - stats['monthly_crime_min']

    df = pd.merge(df, stats, on='geo_grid', how='left')
    print("  添加: monthly_crime_mean, monthly_crime_std, monthly_crime_max, monthly_crime_range")
    return df


# ── 分类特征编码 ──────────────────────────────────────────────────────────────

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    对仍为字符串的分类特征进行编码：
      - 唯一值 < 15 → 独热编码（drop_first=True）
      - 唯一值 ≥ 15 → 标签编码

    跳过的列: crime_id, month, geo_grid, location, lsoa_name
    （这些列不作为模型特征）
    """
    print("编码分类特征...")

    SKIP = {'crime_id', 'month', 'geo_grid', 'location', 'lsoa_name',
            'last_outcome_category'}  # last_outcome_category 已在预处理阶段编码
    cat_cols = [c for c in df.select_dtypes(include='object').columns if c not in SKIP]

    for col in cat_cols:
        n = df[col].nunique()
        if n < 15:
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df.drop(col, axis=1, inplace=True)
            print(f"  {col}: 独热编码（{n} 类）")
        else:
            df[f'{col}_encoded'] = df[col].factorize()[0]
            print(f"  {col}: 标签编码（{n} 类）")

    return df


# ── 特征矩阵 ──────────────────────────────────────────────────────────────────

def create_feature_matrix(df: pd.DataFrame, target_col: str = None):
    """
    创建特征矩阵 X 和目标变量 y（如指定）。
    排除标识类列和目标列。
    """
    print("创建特征矩阵...")

    EXCLUDE = {
        'crime_id', 'location', 'lsoa_name', 'lsoa_code', 'lsoa_prefix',
        'crime_type', 'region', 'month', 'geo_grid', 'last_outcome_category',
    }
    if target_col:
        EXCLUDE.add(target_col)

    drop_cols = [c for c in EXCLUDE if c in df.columns]
    X = df.drop(drop_cols, axis=1, errors='ignore')

    # 移除剩余 object 列
    obj = X.select_dtypes(include='object').columns.tolist()
    if obj:
        print(f"  移除剩余 object 列: {obj}")
        X = X.drop(obj, axis=1)

    y = df[target_col].values if (target_col and target_col in df.columns) else None

    print(f"  特征矩阵: {X.shape[0]:,} 行  x  {X.shape[1]} 列")
    return X, y, X.columns.tolist()


def normalize_features(X: pd.DataFrame):
    """标准化数值特征（需要 sklearn；若不可用则保存原始特征供用户在本机标准化）"""
    print("标准化特征...")
    X_norm = X.copy()
    num_cols = X_norm.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    for col in num_cols:
        if X_norm[col].isna().any():
            X_norm[col].fillna(X_norm[col].mean(), inplace=True)

    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_norm[num_cols] = scaler.fit_transform(X_norm[num_cols])
        print(f"  已用 StandardScaler 标准化 {len(num_cols)} 个特征")
    else:
        # 手动 z-score 标准化（pandas 原生）
        for col in num_cols:
            mu, sigma = X_norm[col].mean(), X_norm[col].std()
            if sigma > 0:
                X_norm[col] = (X_norm[col] - mu) / sigma
        scaler = None
        print(f"  已用 pandas z-score 标准化 {len(num_cols)} 个特征（sklearn 不可用）")

    return X_norm, scaler


def save_features(X, y, feature_names: list, output_dir: str = './'):
    """保存特征矩阵和目标变量"""
    os.makedirs(output_dir, exist_ok=True)
    vals = X.values if hasattr(X, 'values') else X
    np.save(os.path.join(output_dir, 'X_features.npy'), vals)
    if y is not None:
        np.save(os.path.join(output_dir, 'y_target.npy'), y)
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"  特征已保存至: {output_dir}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='犯罪数据特征工程')
    parser.add_argument('--input',  type=str, default='processed_crime_data.csv')
    parser.add_argument('--output', type=str, default='featured_crime_data.csv')
    parser.add_argument('--target', type=str, default=None,
                        help='目标变量列名（通常由 model_training.py 指定）')
    args = parser.parse_args()

    print("=" * 70)
    print(f"特征工程模块 | 输入: {args.input}")
    print("=" * 70)

    if not os.path.exists(args.input):
        print(f"错误: {args.input} 不存在，请先运行 data_preprocessing.py")
        return

    # 1. 加载
    df = load_data(args.input)

    # 2. 空间特征
    print("\n步骤 1: 地理编码特征")
    df = add_geocoding_features(df)

    # 3. 犯罪密度（含目标变量 total_crimes）
    print("\n步骤 2: 犯罪密度特征（含 total_crimes）")
    df = add_crime_density_features(df)

    # 4. 时间聚合特征
    print("\n步骤 3: 时间聚合特征")
    df = add_temporal_aggregated_features(df)

    # 5. 分类特征编码
    print("\n步骤 4: 分类特征编码")
    df = encode_categorical_features(df)

    # 6. 保存完整特征数据
    df.to_csv(args.output, index=False)
    print(f"\n特征数据已保存: {args.output}  ({df.shape[0]:,} x {df.shape[1]})")

    # 7. 创建特征矩阵并标准化
    print("\n步骤 5: 创建特征矩阵和标准化")
    X, y, feature_names = create_feature_matrix(df, args.target)
    X_norm, scaler = normalize_features(X)
    save_features(X_norm, y, feature_names)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n特征工程完成！共生成 {len(feature_names)} 个特征")
    print(f"全部特征列: {feature_names}")


if __name__ == '__main__':
    main()
