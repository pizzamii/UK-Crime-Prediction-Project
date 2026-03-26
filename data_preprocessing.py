"""
数据预处理模块
==============
直接从 dataset/ 文件夹（或 raw_crime_data.csv）读取数据，完成：
  1. 数据加载与基本探索
  2. 缺失值处理、异常值过滤（英国经纬度范围）
  3. 字段标准化（小写/去空格）
  4. 真实时间特征提取：
       - year, month_num, season
       - 周期性编码（month_sin / month_cos）
       - 月内工作日/周末天数
       - 英国银行假日计数与标记
  5. last_outcome_category 编码
  6. 可视化犯罪分布（类型、地区、月度趋势、季节）
  7. 保存 processed_crime_data.csv

输入（两种方式，优先使用 dataset_dir）:
  --dataset_dir dataset/   ← 直接从月份子目录读取（推荐）
  --input raw_crime_data.csv ← 已合并的单一 CSV（向下兼容）
输出:  processed_crime_data.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import calendar
import argparse

try:
    import seaborn as sns
    HAS_SNS = True
except ImportError:
    HAS_SNS = False

# ── 列名映射（data.police.uk 原始列名 → 内部列名）────────────────────────────
COL_MAP = {
    'Crime ID':               'crime_id',
    'Month':                  'month',
    'Reported by':            'region',
    'Falls within':           'falls_within',
    'Longitude':              'longitude',
    'Latitude':               'latitude',
    'Location':               'location',
    'LSOA code':              'lsoa_code',
    'LSOA name':              'lsoa_name',
    'Crime type':             'crime_type',
    'Last outcome category':  'last_outcome_category',
    'Context':                'context',
}

# ── 英国银行假日（Bank Holidays）完整表 ──────────────────────────────────────
UK_BANK_HOLIDAYS = {
    2019: ['2019-01-01', '2019-04-19', '2019-04-22', '2019-05-06',
           '2019-05-27', '2019-08-26', '2019-12-25', '2019-12-26'],
    2020: ['2020-01-01', '2020-04-10', '2020-04-13', '2020-05-08',
           '2020-05-25', '2020-08-31', '2020-12-25', '2020-12-28'],
    2021: ['2021-01-01', '2021-04-02', '2021-04-05', '2021-05-03',
           '2021-05-31', '2021-08-30', '2021-12-27', '2021-12-28'],
    2022: ['2022-01-03', '2022-04-15', '2022-04-18', '2022-05-02',
           '2022-06-02', '2022-06-03', '2022-08-29', '2022-12-26', '2022-12-27'],
    2023: ['2023-01-02', '2023-04-07', '2023-04-10', '2023-05-01',
           '2023-05-08', '2023-05-29', '2023-08-28', '2023-12-25', '2023-12-26'],
    2024: ['2024-01-01', '2024-03-29', '2024-04-01', '2024-05-06',
           '2024-05-27', '2024-08-26', '2024-12-25', '2024-12-26'],
    2025: ['2025-01-01', '2025-04-18', '2025-04-21', '2025-05-05',
           '2025-05-26', '2025-08-25', '2025-12-25', '2025-12-26'],
}

# ── 季节映射 ──────────────────────────────────────────────────────────────────
SEASON_MAP = {3: 1, 4: 1, 5: 1,    # Spring
              6: 2, 7: 2, 8: 2,    # Summer
              9: 3, 10: 3, 11: 3,  # Autumn
              12: 4, 1: 4, 2: 4}   # Winter

SEASON_LABELS = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}


# ── 从 dataset/ 目录加载数据 ─────────────────────────────────────────────────

def load_from_dataset_dir(dataset_dir: str) -> pd.DataFrame:
    """
    直接从 dataset/ 文件夹加载所有月份的街道犯罪数据。

    目录结构示例:
        dataset/
          2024-05/
            2024-05-avon-and-somerset-street.csv
            2024-05-bedfordshire-street.csv
            ...
          2024-06/
            ...
          2025-04/
            ...

    会自动递归搜索所有 *-street.csv 文件并合并。
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"dataset 目录 '{dataset_dir}' 不存在。\n"
            f"请确认路径正确，或使用 --input raw_crime_data.csv 指定已合并的 CSV 文件。"
        )

    # 搜索所有 street CSV 文件（递归两级）
    pattern1 = os.path.join(dataset_dir, '*', '*-street.csv')
    pattern2 = os.path.join(dataset_dir, '*-street.csv')
    files = sorted(glob.glob(pattern1) + glob.glob(pattern2))

    if not files:
        raise FileNotFoundError(
            f"在 '{dataset_dir}' 下未找到任何 *-street.csv 文件。\n"
            f"请确认目录结构：dataset/<YYYY-MM>/<YYYY-MM>-<force>-street.csv"
        )

    print(f"找到 {len(files)} 个 street CSV 文件，开始加载...")

    chunks = []
    total_rows = 0
    for i, fpath in enumerate(files, 1):
        try:
            chunk = pd.read_csv(fpath, dtype={'Crime ID': str}, low_memory=False)
            # 重命名列
            chunk.rename(columns=COL_MAP, inplace=True)
            # 只保留需要的列
            keep = [v for v in COL_MAP.values() if v in chunk.columns]
            chunk = chunk[keep]
            chunks.append(chunk)
            total_rows += len(chunk)
            if i % 50 == 0 or i == len(files):
                print(f"  [{i}/{len(files)}] 已加载 {total_rows:,} 行...")
        except Exception as e:
            print(f"  警告: 跳过文件 {os.path.basename(fpath)}: {e}")

    if not chunks:
        raise ValueError("没有成功加载任何数据文件，请检查文件格式。")

    print(f"\n合并所有数据...")
    df = pd.concat(chunks, ignore_index=True)
    print(f"  合并完成: {df.shape[0]:,} 行  x  {df.shape[1]} 列")

    # 显示月份覆盖范围
    if 'month' in df.columns:
        months = sorted(df['month'].dropna().unique())
        print(f"  月份范围: {months[0]} ~ {months[-1]}  (共 {len(months)} 个月)")

    return df


# ── 从单一 CSV 文件加载数据（向下兼容）────────────────────────────────────────

def load_from_csv(file_path: str) -> pd.DataFrame:
    """加载已合并的单一 CSV（如 raw_crime_data.csv）"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"输入文件 '{file_path}' 不存在。\n"
            f"请使用 --dataset_dir dataset/ 直接从月份目录读取，或先运行 data_download.py。"
        )
    print(f"加载数据: {file_path}")
    df = pd.read_csv(file_path, dtype={'crime_id': str}, low_memory=False)
    print(f"  形状: {df.shape[0]:,} 行  x  {df.shape[1]} 列")
    return df


def basic_exploration(df: pd.DataFrame):
    """基本数据探索"""
    print("\n" + "=" * 60)
    print("数据概览")
    print("=" * 60)
    print(f"形状: {df.shape}")
    print(f"\n列类型:\n{df.dtypes}")
    print(f"\n缺失值:\n{df.isnull().sum()}")
    print(f"\n字符串列唯一值数:")
    for col in df.select_dtypes(include='object').columns:
        print(f"  {col:30s}: {df[col].nunique():>6,} 个唯一值")
    if 'month' in df.columns:
        print(f"\n月份范围: {df['month'].min()} ~ {df['month'].max()}")
        print(f"月份数量: {df['month'].nunique()}")


# ── 数据清洗 ─────────────────────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """数据清洗"""
    print("\n开始数据清洗...")
    df = df.copy()

    # 1. 验证 month 列存在
    if 'month' not in df.columns:
        raise ValueError(
            "数据中缺少 'month' 列！\n"
            "data.police.uk 的街道犯罪数据包含 'Month' 列（格式：YYYY-MM）。\n"
            "请确认使用的是正确的 *-street.csv 文件。"
        )

    # 2. 验证 month 格式为 YYYY-MM
    month_sample = df['month'].dropna().head(5).tolist()
    print(f"  month 列样本: {month_sample}")
    invalid_months = df['month'].dropna()[~df['month'].dropna().str.match(r'^\d{4}-\d{2}$')]
    if len(invalid_months) > 0:
        print(f"  警告: {len(invalid_months)} 条月份格式异常，已删除")
        df = df[df['month'].str.match(r'^\d{4}-\d{2}$', na=False)]

    # 3. 处理 crime_id 缺失（Anti-social behaviour 无 crime_id）
    df['crime_id'] = df['crime_id'].fillna('').astype(str)

    # 4. 数值类型转换
    for col in ['longitude', 'latitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 5. 删除经纬度缺失行
    before = len(df)
    df = df.dropna(subset=['longitude', 'latitude'])
    print(f"  删除经纬度缺失: {before:,} → {len(df):,}")

    # 6. 英国地理范围过滤
    before = len(df)
    df = df[
        (df['longitude'] >= -8.0) & (df['longitude'] <= 2.0) &
        (df['latitude'] >= 49.0) & (df['latitude'] <= 61.0)
    ]
    print(f"  UK 范围过滤: {before:,} → {len(df):,}")

    # 7. 字符串列统一格式（小写、去空格），保留 month 列
    str_cols = df.select_dtypes(include='object').columns.tolist()
    for col in str_cols:
        if col == 'month':
            continue
        df[col] = df[col].astype(str).str.strip().str.lower()
        df[col] = df[col].replace('nan', np.nan)

    # 8. 填充字符串列缺失值
    for col in ['location', 'last_outcome_category', 'lsoa_name', 'lsoa_code']:
        if col in df.columns:
            df[col] = df[col].fillna('unknown')

    # 9. region / crime_type 不能为空
    if 'region' in df.columns:
        df = df[df['region'].notna() & (df['region'] != 'nan') & (df['region'] != '')]
    if 'crime_type' in df.columns:
        df = df[df['crime_type'].notna() & (df['crime_type'] != 'nan') & (df['crime_type'] != '')]

    # 10. LSOA 前缀
    if 'lsoa_code' in df.columns:
        df['lsoa_prefix'] = df['lsoa_code'].str[:3]

    print(f"  清洗完成，剩余 {len(df):,} 条记录")
    return df


# ── 时间特征提取 ──────────────────────────────────────────────────────────────

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    从真实 Month 列（YYYY-MM）提取时间特征：
      - year, month_num
      - season (1=Spring, 2=Summer, 3=Autumn, 4=Winter)
      - month_sin, month_cos （周期性编码）
      - n_weekdays, n_weekends （月内工作日/周末天数）
      - n_bank_holidays, is_holiday_month
    """
    print("\n提取时间特征（基于真实 Month 列）...")

    df = df.copy()
    df['year']      = df['month'].str[:4].astype(int)
    df['month_num'] = df['month'].str[5:7].astype(int)
    df['season']    = df['month_num'].map(SEASON_MAP)

    # 周期性编码（sin/cos 变换，保留月份周期连续性）
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)

    # ── 每月工作日 / 周末天数（按 (year, month_num) 唯一组合计算） ──
    ym_unique = df[['year', 'month_num']].drop_duplicates().copy()

    def _weekday_weekend(row):
        y, m = int(row['year']), int(row['month_num'])
        total = calendar.monthrange(y, m)[1]
        wd = sum(1 for d in range(1, total + 1) if calendar.weekday(y, m, d) < 5)
        return pd.Series({'n_weekdays': wd, 'n_weekends': total - wd})

    ym_unique[['n_weekdays', 'n_weekends']] = ym_unique.apply(_weekday_weekend, axis=1)
    df = df.merge(ym_unique, on=['year', 'month_num'], how='left')

    # ── 银行假日 ──
    def _bank_holidays(row):
        y, m = int(row['year']), int(row['month_num'])
        holidays = UK_BANK_HOLIDAYS.get(y, [])
        cnt = sum(1 for h in holidays if int(h.split('-')[1]) == m)
        return pd.Series({'n_bank_holidays': cnt, 'is_holiday_month': int(cnt > 0)})

    bh_unique = df[['year', 'month_num']].drop_duplicates().copy()
    bh_unique[['n_bank_holidays', 'is_holiday_month']] = bh_unique.apply(_bank_holidays, axis=1)
    df = df.merge(bh_unique, on=['year', 'month_num'], how='left')

    print(f"  新增特征: year, month_num, season, month_sin, month_cos, "
          f"n_weekdays, n_weekends, n_bank_holidays, is_holiday_month")
    return df


# ── Last Outcome Category 编码 ────────────────────────────────────────────────

def encode_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    对 last_outcome_category 进行标签编码，生成数值特征 last_outcome_encoded。
    保留原始字符串列供可视化使用。使用 pd.factorize() 避免 sklearn 依赖。
    """
    if 'last_outcome_category' not in df.columns:
        return df

    print("\n编码 last_outcome_category ...")
    codes, uniques = pd.factorize(df['last_outcome_category'].fillna('unknown'))
    df['last_outcome_encoded'] = codes
    print(f"  共 {len(uniques)} 种结果类别")
    top5 = df['last_outcome_category'].value_counts().head(5)
    for cat, cnt in top5.items():
        print(f"    {str(cat):45s}: {cnt:>8,}")
    return df


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def visualize_distributions(df: pd.DataFrame, output_dir: str = './'):
    """生成犯罪分布可视化图表"""
    print("\n生成可视化图表...")

    # 1. 犯罪类型分布
    fig, ax = plt.subplots(figsize=(14, 6))
    ct = df['crime_type'].value_counts()
    ax.bar(ct.index, ct.values, color='steelblue', edgecolor='white')
    ax.set_title('Crime Type Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crime_distribution.png'), dpi=150)
    plt.close()
    print("  已保存 crime_distribution.png")

    # 2. 地区分布（Top 15）
    if 'region' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        rc = df['region'].value_counts().head(15)
        ax.barh(rc.index[::-1], rc.values[::-1], color='salmon', edgecolor='white')
        ax.set_title('Top 15 Regions by Crime Count', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'region_distribution.png'), dpi=150)
        plt.close()
        print("  已保存 region_distribution.png")

    # 3. 月度犯罪趋势（真实时间轴）
    if 'month' in df.columns:
        fig, ax = plt.subplots(figsize=(16, 5))
        mc = df.groupby('month').size().sort_index()
        ax.plot(range(len(mc)), mc.values, marker='o', linewidth=1.5,
                markersize=4, color='steelblue')
        step = max(1, len(mc) // 12)
        ax.set_xticks(range(0, len(mc), step))
        ax.set_xticklabels([mc.index[i] for i in range(0, len(mc), step)],
                           rotation=45, ha='right')
        ax.set_title('Monthly Crime Trend (Real Data)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Crime Count')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'monthly_crime_trend.png'), dpi=150)
        plt.close()
        print("  已保存 monthly_crime_trend.png")

    # 4. 季节分布
    if 'season' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = df.groupby('season').size()
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
        labels = [SEASON_LABELS.get(s, str(s)) for s in sc.index]
        ax.bar(labels, sc.values, color=colors[:len(sc)], edgecolor='white')
        ax.set_title('Crime Distribution by Season', fontsize=14, fontweight='bold')
        ax.set_ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'seasonal_crime_distribution.png'), dpi=150)
        plt.close()
        print("  已保存 seasonal_crime_distribution.png")

    # 5. Last Outcome Category（Top 10）
    if 'last_outcome_category' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        oc = df['last_outcome_category'].value_counts().head(10)
        ax.barh(oc.index[::-1], oc.values[::-1], color='mediumpurple', edgecolor='white')
        ax.set_title('Top 10 Last Outcome Categories', fontsize=14, fontweight='bold')
        ax.set_xlabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'outcome_distribution.png'), dpi=150)
        plt.close()
        print("  已保存 outcome_distribution.png")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """完整的预处理流程"""
    df = clean_data(df)
    df = extract_time_features(df)
    df = encode_outcome(df)
    return df


def main():
    parser = argparse.ArgumentParser(description='UK 犯罪数据预处理')
    parser.add_argument(
        '--dataset_dir', type=str, default='dataset',
        help='dataset 目录路径，包含月份子文件夹（默认: dataset/）。'
             '优先于 --input 使用。'
    )
    parser.add_argument(
        '--input', type=str, default='raw_crime_data.csv',
        help='已合并的单一 CSV 文件（向下兼容；当 --dataset_dir 不存在时使用）'
    )
    parser.add_argument(
        '--output', type=str, default='processed_crime_data.csv',
        help='输出文件（默认: processed_crime_data.csv）'
    )
    args = parser.parse_args()

    print("=" * 70)
    print("数据预处理模块")
    print("=" * 70)

    # ── 加载数据：优先使用 dataset_dir，其次 input CSV ──
    if os.path.isdir(args.dataset_dir):
        print(f"\n[数据源] dataset 目录: {args.dataset_dir}")
        df = load_from_dataset_dir(args.dataset_dir)
    elif os.path.exists(args.input):
        print(f"\n[数据源] 单一 CSV 文件: {args.input}")
        df = load_from_csv(args.input)
    else:
        print(f"\n错误: 既找不到 dataset 目录 '{args.dataset_dir}'，")
        print(f"      也找不到输入文件 '{args.input}'。")
        print(f"\n请将从 data.police.uk 下载的月份数据放入 dataset/ 文件夹：")
        print(f"    dataset/")
        print(f"    ├── 2024-05/")
        print(f"    │   ├── 2024-05-avon-and-somerset-street.csv")
        print(f"    │   └── ...")
        print(f"    └── 2025-04/")
        return

    # ── 探索 ──
    basic_exploration(df)

    # ── 预处理（清洗 + 时间特征 + 编码）──
    df = preprocess(df)

    # ── 可视化 ──
    visualize_distributions(df)

    # ── 保存 ──
    df.to_csv(args.output, index=False)
    print(f"\n预处理完成! 已保存 → {args.output}")
    print(f"  形状: {df.shape[0]:,} 行  x  {df.shape[1]} 列")
    print(f"  月份范围: {df['month'].min()} ~ {df['month'].max()}")
    print(f"  犯罪类型: {df['crime_type'].nunique()} 种")
    print(f"  地区数量: {df['region'].nunique() if 'region' in df.columns else 'N/A'}")
    print(f"  列: {list(df.columns)}")


if __name__ == '__main__':
    main()
