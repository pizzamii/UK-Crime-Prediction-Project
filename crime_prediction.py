"""
犯罪预测模块
============
使用训练好的模型对每个 geo_grid 网格进行犯罪数量预测，
生成预测排名表（crime_predictions.csv）和热力图数据（heatmap_data.csv）。

流程：
  1. 加载调优后的 XGBoost 模型（优先）或 Random Forest
  2. 加载 core_feature_names.pkl 和 scaler.pkl
  3. 从 featured_crime_data.csv 聚合网格级特征
  4. 标准化 → 预测 → 计算置信区间（Bootstrap）
  5. 保存预测结果和热力图数据
  6. 生成预测热点排名图

输入:  featured_crime_data.csv, tuned_xgboost_model.pkl,
       core_feature_names.pkl, scaler.pkl
输出:  crime_predictions.csv, heatmap_data.csv,
       predicted_total_crimes_top20.png
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import StandardScaler
import traceback


# ── 加载工具函数 ──────────────────────────────────────────────────────────────

def _load_pkl(path: str, label: str):
    if not os.path.exists(path):
        print(f"  [missing] {label}: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print(f"  [ok] {label} 已加载: {path}")
        return obj
    except Exception as e:
        print(f"  [error] 加载 {label} 失败: {e}")
        return None


def load_model(model_path: str):
    """优先加载指定模型，失败时尝试备选模型"""
    model = _load_pkl(model_path, 'Model')
    if model is None:
        fallbacks = [
            'tuned_xgboost_model.pkl',
            'tuned_random_forest_model.pkl',
        ]
        for fb in fallbacks:
            if fb != model_path and os.path.exists(fb):
                model = _load_pkl(fb, f'备选模型 ({fb})')
                if model:
                    break
    return model


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据文件不存在: {path}")
    df = pd.read_csv(path)
    print(f"  数据加载: {df.shape[0]:,} 行  x  {df.shape[1]} 列")
    return df


# ── 预测数据准备 ──────────────────────────────────────────────────────────────

def prepare_prediction_data(df: pd.DataFrame, core_features: list):
    """
    按 geo_grid 聚合数值列（取均值），准备预测特征矩阵。
    缺少的特征列用 0 填充。
    """
    print(f"\n准备预测数据（{len(core_features)} 个核心特征）...")

    # 找区域标识列
    region_col = next((c for c in ['geo_grid', 'lsoa_code', 'region']
                       if c in df.columns), None)
    if region_col is None:
        raise ValueError("数据中缺少区域标识列（geo_grid / lsoa_code / region）")
    print(f"  区域标识列: {region_col}")

    num_cols  = df.select_dtypes(include='number').columns.tolist()
    agg_df    = df.groupby(region_col)[num_cols].mean().reset_index()

    # 补齐缺失特征
    missing = [f for f in core_features if f not in agg_df.columns]
    if missing:
        print(f"  补零的缺失特征: {missing}")
        for f in missing:
            agg_df[f] = 0.0

    X       = agg_df[core_features].copy()
    regions = agg_df[region_col]
    print(f"  特征矩阵: {X.shape[0]:,} 网格  x  {X.shape[1]} 特征")
    return X, regions


# ── 置信区间（Bootstrap）──────────────────────────────────────────────────────

def bootstrap_confidence_intervals(model, X_arr: np.ndarray,
                                   n_bootstraps: int = 200, confidence: float = 0.95):
    """用 Bootstrap 方法估计预测置信区间"""
    print(f"  Bootstrap 置信区间（{n_bootstraps} 次）...")
    preds = np.zeros((X_arr.shape[0], n_bootstraps))
    for i in range(n_bootstraps):
        idx = np.random.choice(X_arr.shape[0], X_arr.shape[0], replace=True)
        preds[:, i] = model.predict(X_arr[idx])

    alpha = (1 - confidence) / 2
    lower = np.percentile(preds, alpha * 100, axis=1)
    upper = np.percentile(preds, (1 - alpha) * 100, axis=1)
    return lower, upper


# ── 可视化 ─────────────────────────────────────────────────────────────────────

def visualize_predictions(pred_df: pd.DataFrame, top_n: int = 20,
                           target: str = 'total_crimes', output_dir: str = './'):
    """绘制 Top-N 犯罪热点排名图（含置信区间误差条）"""
    top = pred_df.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(top))

    bars = ax.barh(list(y_pos), top['Predicted_Crime'].values,
                   color=plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(top)))[::-1],
                   edgecolor='white', height=0.6)

    # 误差条
    if 'Lower_Bound' in top.columns and 'Upper_Bound' in top.columns:
        for i, (_, row) in enumerate(top.iterrows()):
            lo = max(0, row['Predicted_Crime'] - row['Lower_Bound'])
            hi = max(0, row['Upper_Bound'] - row['Predicted_Crime'])
            ax.errorbar(
                x=row['Predicted_Crime'], y=i,
                xerr=[[lo], [hi]],
                fmt='none', ecolor='gray', capsize=3, linewidth=1.2
            )

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(top['Region'].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Predicted Crime Count')
    ax.set_title(f'Top {top_n} Predicted Crime Hotspots', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    out = os.path.join(output_dir, f'predicted_{target}_top{top_n}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存 {out}")


def create_heatmap_data(pred_df: pd.DataFrame, geo_df: pd.DataFrame,
                        region_col: str = 'geo_grid',
                        output_path: str = 'heatmap_data.csv'):
    """
    将预测结果与经纬度合并，生成热力图 CSV。
    geo_df 须包含 latitude, longitude, geo_grid 列。
    """
    print("\n准备热力图数据...")

    if not all(c in geo_df.columns for c in ['latitude', 'longitude', region_col]):
        print(f"  缺少必要列，跳过热力图生成")
        return None

    coords = geo_df.groupby(region_col)[['latitude', 'longitude']].mean().reset_index()
    merged = pred_df.merge(coords, left_on='Region', right_on=region_col, how='inner')

    if merged.empty:
        print("  无法匹配预测结果与坐标，跳过")
        return None

    heatmap = merged[['Region', 'latitude', 'longitude', 'Predicted_Crime']].copy()
    heatmap.rename(columns={'Predicted_Crime': 'predicted_crimes'}, inplace=True)
    heatmap.to_csv(output_path, index=False)
    print(f"  热力图数据已保存: {output_path}  ({len(heatmap):,} 条)")
    return heatmap


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='犯罪预测（基于训练好的模型）')
    parser.add_argument('--input',   type=str, default='featured_crime_data.csv')
    parser.add_argument('--model',   type=str, default='tuned_xgboost_model.pkl')
    parser.add_argument('--target',  type=str, default='total_crimes')
    parser.add_argument('--output',  type=str, default='crime_predictions.csv')
    parser.add_argument('--heatmap', type=str, default='heatmap_data.csv')
    parser.add_argument('--top_n',   type=int, default=20)
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("犯罪预测模块")
    print(f"{'='*70}\n")

    # 1. 加载模型
    model = load_model(args.model)
    if model is None:
        print("错误: 无法加载任何模型，请先运行 model_training.py")
        sys.exit(1)

    # 2. 加载组件
    scaler       = _load_pkl('scaler.pkl', 'Scaler')
    core_feats   = _load_pkl('core_feature_names.pkl', 'Core Features')
    if core_feats is None:
        core_feats = _load_pkl('feature_names.pkl', 'Feature Names')
    if core_feats is None:
        print("错误: 无法加载特征名称")
        sys.exit(1)

    print(f"\n  使用特征 ({len(core_feats)}): {core_feats}")

    # 3. 加载数据
    df = load_data(args.input)

    # 4. 准备预测特征
    X, regions = prepare_prediction_data(df, core_feats)

    # 5. 标准化
    if scaler is not None:
        try:
            X_norm = scaler.transform(X)
        except Exception as e:
            print(f"  标准化出错（{e}），使用新 StandardScaler")
            X_norm = StandardScaler().fit_transform(X)
    else:
        X_norm = StandardScaler().fit_transform(X)

    # 6. 预测
    try:
        print("\n进行预测...")
        preds = model.predict(X_norm)

        # 置信区间
        try:
            lower, upper = bootstrap_confidence_intervals(model, X_norm)
        except Exception:
            lower = preds * 0.85
            upper = preds * 1.15
            print("  使用简单估计置信区间 (±15%)")

        results_df = pd.DataFrame({
            'Region':          regions.values,
            'Predicted_Crime': preds,
            'Lower_Bound':     lower,
            'Upper_Bound':     upper,
        }).sort_values('Predicted_Crime', ascending=False).reset_index(drop=True)

        # 保存
        results_df.to_csv(args.output, index=False)
        print(f"\n  预测结果已保存: {args.output}  ({len(results_df):,} 条)")

        # 可视化
        visualize_predictions(results_df, top_n=args.top_n, target=args.target)

        # 热力图数据
        if all(c in df.columns for c in ['latitude', 'longitude', 'geo_grid']):
            create_heatmap_data(results_df, df, 'geo_grid', args.heatmap)

        # 输出 Top-5
        print(f"\nTop-{min(5, len(results_df))} 犯罪热点:")
        for _, row in results_df.head(5).iterrows():
            print(f"  {row['Region']:20s}  预测: {row['Predicted_Crime']:8.1f}"
                  f"  (95% CI: {row['Lower_Bound']:.1f} ~ {row['Upper_Bound']:.1f})")

    except Exception as e:
        print(f"预测出错: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
