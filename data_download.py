"""
UK Police Crime Data Downloader
================================
从 data.police.uk 下载街道级别犯罪数据（street crimes），
提取真实的 Month 列及 Last outcome category 等字段，
生成供后续 pipeline 使用的 raw_crime_data.csv。

使用方法：
    python data_download.py
    python data_download.py --start 2021-01 --end 2023-12
    python data_download.py --start 2020-01 --end 2023-12 --output raw_crime_data.csv

说明：
    - 数据来源: https://data.police.uk/data/
    - 每月归档 ZIP 文件约 50-100 MB，全量下载需要较长时间
    - 下载中途中断后可重启，已下载的月份会自动跳过
    - 最终输出 raw_crime_data.csv，包含 Month 列（真实日期）
"""

import os
import sys
import zipfile
import io
import glob
import argparse
import time
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

import pandas as pd
import numpy as np
import requests

# ── 配置 ─────────────────────────────────────────────────────────────────────
ARCHIVE_URL = "https://data.police.uk/data/archive/{year}-{month:02d}.zip"

# 保留的列（来自 data.police.uk 原始字段）
KEEP_COLS = {
    'Crime ID':                'crime_id',
    'Month':                   'month',
    'Falls within':            'region',
    'Longitude':               'longitude',
    'Latitude':                'latitude',
    'Location':                'location',
    'LSOA code':               'lsoa_code',
    'LSOA name':               'lsoa_name',
    'Crime type':              'crime_type',
    'Last outcome category':   'last_outcome_category',
}

DEFAULT_START = '2020-01'
DEFAULT_END   = '2023-12'


# ── 辅助函数 ──────────────────────────────────────────────────────────────────

def parse_ym(ym_str: str):
    """解析 YYYY-MM 格式字符串"""
    try:
        return datetime.strptime(ym_str, '%Y-%m').date().replace(day=1)
    except ValueError:
        raise ValueError(f"日期格式错误: {ym_str}，应为 YYYY-MM")


def month_range(start: str, end: str):
    """生成从 start 到 end (含) 的月份列表"""
    cur = parse_ym(start)
    fin = parse_ym(end)
    months = []
    while cur <= fin:
        months.append((cur.year, cur.month))
        cur += relativedelta(months=1)
    return months


def download_month(year: int, month: int, cache_dir: str, session: requests.Session):
    """
    下载单个月份的归档 ZIP 并返回该月所有街道犯罪数据。
    已下载的月份文件会被缓存，避免重复下载。
    """
    cache_path = os.path.join(cache_dir, f"{year}-{month:02d}.zip")
    url = ARCHIVE_URL.format(year=year, month=month)

    # 如果缓存已存在，直接读取
    if os.path.exists(cache_path):
        print(f"    [cache] 已有 {year}-{month:02d}.zip，跳过下载")
        zip_bytes = open(cache_path, 'rb').read()
    else:
        print(f"    [fetch] {url}")
        try:
            resp = session.get(url, timeout=120)
            resp.raise_for_status()
            zip_bytes = resp.content
            with open(cache_path, 'wb') as f:
                f.write(zip_bytes)
            print(f"    [ok]    已保存 ({len(zip_bytes)/1024/1024:.1f} MB)")
        except requests.RequestException as e:
            print(f"    [error] 下载失败: {e}")
            return None

    # 解压并读取所有 *-street.csv
    frames = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            street_files = [n for n in zf.namelist() if n.endswith('-street.csv')]
            for fname in street_files:
                try:
                    with zf.open(fname) as f:
                        df = pd.read_csv(f, dtype=str)
                    frames.append(df)
                except Exception as e:
                    print(f"      [warn] 读取 {fname} 失败: {e}")
    except zipfile.BadZipFile as e:
        print(f"    [error] 无效 ZIP 文件: {e}")
        # 删除损坏的缓存
        os.remove(cache_path)
        return None

    if not frames:
        return None

    month_df = pd.concat(frames, ignore_index=True)
    return month_df


def process_raw_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始数据做列重命名和基本清洗：
    - 只保留 KEEP_COLS 中定义的列
    - 重命名为统一的小写列名
    - 删除经纬度缺失的行
    """
    # 只保留存在的列
    available = {k: v for k, v in KEEP_COLS.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # 转换数值类型
    for col in ['longitude', 'latitude']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 删除经纬度缺失的行（无法用于空间分析）
    df = df.dropna(subset=['longitude', 'latitude'])

    return df


def main():
    parser = argparse.ArgumentParser(
        description='从 data.police.uk 下载英国街道犯罪数据',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--start', type=str, default=DEFAULT_START,
                        help=f'起始月份 YYYY-MM（默认: {DEFAULT_START}）')
    parser.add_argument('--end',   type=str, default=DEFAULT_END,
                        help=f'结束月份 YYYY-MM（默认: {DEFAULT_END}）')
    parser.add_argument('--output', type=str, default='raw_crime_data.csv',
                        help='输出 CSV 文件路径（默认: raw_crime_data.csv）')
    parser.add_argument('--cache_dir', type=str, default='_zip_cache',
                        help='ZIP 缓存目录（默认: _zip_cache）')
    parser.add_argument('--max_rows', type=int, default=None,
                        help='最多保留的行数（调试用，None=全量）')
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    months = month_range(args.start, args.end)
    print(f"\n{'='*70}")
    print(f"UK Police Crime Data Downloader")
    print(f"时间范围: {args.start} ~ {args.end}  ({len(months)} 个月)")
    print(f"输出文件: {args.output}")
    print(f"缓存目录: {args.cache_dir}")
    print(f"{'='*70}\n")

    session = requests.Session()
    session.headers.update({'User-Agent': 'UK-Crime-Research/1.0'})

    all_frames = []
    total_rows = 0

    for i, (year, month) in enumerate(months, 1):
        print(f"\n[{i}/{len(months)}] 处理 {year}-{month:02d} ...")
        t0 = time.time()

        df = download_month(year, month, args.cache_dir, session)
        if df is None:
            print(f"    [skip] {year}-{month:02d} 跳过")
            continue

        df = process_raw_df(df)
        n = len(df)
        total_rows += n
        all_frames.append(df)

        elapsed = time.time() - t0
        print(f"    [done] {n:,} 条记录，耗时 {elapsed:.1f}s，累计 {total_rows:,} 条")

        # 调试模式：行数达到上限时停止
        if args.max_rows and total_rows >= args.max_rows:
            print(f"\n已达到最大行数限制 {args.max_rows}，停止下载")
            break

    if not all_frames:
        print("\n错误: 未能下载任何数据，请检查网络连接或日期范围")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"合并数据...")
    combined = pd.concat(all_frames, ignore_index=True)

    # 如果设置了 max_rows，截断
    if args.max_rows:
        combined = combined.head(args.max_rows)

    print(f"总行数: {len(combined):,}")
    print(f"列: {list(combined.columns)}")
    print(f"\n犯罪类型分布:")
    if 'crime_type' in combined.columns:
        top_types = combined['crime_type'].value_counts().head(10)
        for ct, cnt in top_types.items():
            print(f"  {ct:40s}: {cnt:>8,}")

    print(f"\n月份分布:")
    if 'month' in combined.columns:
        month_counts = combined['month'].value_counts().sort_index()
        for m, cnt in month_counts.items():
            print(f"  {m}: {cnt:>8,}")

    # 保存
    print(f"\n保存到 {args.output} ...")
    combined.to_csv(args.output, index=False)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"完成! 文件大小: {size_mb:.1f} MB，共 {len(combined):,} 行")
    print(f"\n下一步: python data_preprocessing.py --input {args.output}")


if __name__ == '__main__':
    main()
