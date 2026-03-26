"""
一键运行完整 Pipeline
=====================
步骤：
  0. 数据预处理       — python data_preprocessing.py  --dataset_dir dataset/
  1. 特征工程         — python feature_engineering.py
  2. 模型训练         — python model_training.py
  3. 犯罪预测         — python crime_prediction.py
  4. (手动) 可视化    — streamlit run visualization_app.py

数据来源：
  将从 data.police.uk 下载的月份数据放入 dataset/ 文件夹：
      dataset/
      ├── 2024-05/
      │   ├── 2024-05-avon-and-somerset-street.csv
      │   └── ...
      └── 2025-04/
          └── ...

使用方法：
    python run_all.py                          # 从 dataset/ 开始完整流程
    python run_all.py --dataset_dir mydata/    # 指定自定义 dataset 路径
    python run_all.py --optuna_trials 30       # 减少贝叶斯优化次数（加快速度）
"""

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime


# ── 依赖检查 ─────────────────────────────────────────────────────────────────

REQUIRED_PACKAGES = [
    'pandas', 'numpy', 'sklearn', 'xgboost',
    'matplotlib', 'seaborn', 'shap', 'optuna',
    'requests', 'dateutil',
]

def check_dependencies():
    print("检查 Python 依赖...")
    pkg_map = {'sklearn': 'scikit-learn', 'dateutil': 'python-dateutil'}
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg_map.get(pkg, pkg))

    if missing:
        print(f"  缺少依赖: {missing}")
        ans = input("  是否自动安装? (y/n): ").strip().lower()
        if ans == 'y':
            subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing, check=True)
        else:
            print("  请手动安装后重试: pip install " + ' '.join(missing))
            sys.exit(1)
    else:
        print("  所有依赖已满足 ✓")


# ── 单步执行 ──────────────────────────────────────────────────────────────────

def run_step(cmd: str, label: str) -> bool:
    """运行单个步骤，打印输出，返回是否成功"""
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            text=True, capture_output=True
        )
        print(result.stdout)
        # 只打印非常规警告
        if result.stderr:
            lines = [l for l in result.stderr.splitlines()
                     if l.strip() and not any(kw in l for kw in
                        ['UserWarning', 'FutureWarning', 'DeprecationWarning',
                         'ConvergenceWarning', 'INFO'])]
            if lines:
                print("  [stderr]", '\n  '.join(lines[:15]))

        print(f"\n  ✓  完成 ({time.time()-t0:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  ✗  失败 ({time.time()-t0:.1f}s)")
        print(f"  错误:\n{e.stderr[:3000]}")
        return False


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='UK 犯罪预测系统 — 全流程一键运行')
    parser.add_argument(
        '--dataset_dir', type=str, default='dataset',
        help='月份数据目录（默认: dataset/），包含 2024-05/ 等子文件夹'
    )
    parser.add_argument('--processed',     type=str, default='processed_crime_data.csv')
    parser.add_argument('--featured',      type=str, default='featured_crime_data.csv')
    parser.add_argument('--target',        type=str, default='total_crimes')
    parser.add_argument('--optuna_trials', type=int, default=50,
                        help='贝叶斯优化 trial 数（默认 50，减少可加快速度）')
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  UK 犯罪预测系统 — 全流程 Pipeline")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*72}")

    # 检查 dataset 目录
    if not os.path.isdir(args.dataset_dir):
        print(f"\n错误: 找不到 dataset 目录 '{args.dataset_dir}'")
        print(f"请将从 data.police.uk 下载的月份数据放入该目录：")
        print(f"    {args.dataset_dir}/")
        print(f"    ├── 2024-05/")
        print(f"    │   ├── 2024-05-avon-and-somerset-street.csv")
        print(f"    │   └── ...")
        print(f"    └── 2025-04/")
        sys.exit(1)

    # 检查依赖
    check_dependencies()

    total_t0 = time.time()
    failed_steps = []

    # ── 步骤 0: 数据预处理 ───────────────────────────────────────────────────
    ok = run_step(
        f"python data_preprocessing.py "
        f"--dataset_dir {args.dataset_dir} "
        f"--output {args.processed}",
        f"步骤 0/3: 数据预处理（直接读取 {args.dataset_dir}/ → {args.processed}）"
    )
    if not ok:
        print("错误: 数据预处理失败，中止")
        sys.exit(1)

    # ── 步骤 1: 特征工程 ─────────────────────────────────────────────────────
    ok = run_step(
        f"python feature_engineering.py "
        f"--input {args.processed} "
        f"--output {args.featured}",
        "步骤 1/3: 特征工程（地理网格 + 犯罪密度 + 时间聚合）"
    )
    if not ok:
        print("错误: 特征工程失败，中止")
        sys.exit(1)

    # ── 步骤 2: 模型训练 ─────────────────────────────────────────────────────
    ok = run_step(
        f"python model_training.py "
        f"--input {args.featured} "
        f"--target {args.target} "
        f"--optuna_trials {args.optuna_trials}",
        "步骤 2/3: 模型训练（7种对比 + 贝叶斯优化 + SHAP）"
    )
    if not ok:
        failed_steps.append("model_training")

    # ── 步骤 3: 犯罪预测 ─────────────────────────────────────────────────────
    ok = run_step(
        f"python crime_prediction.py "
        f"--input {args.featured} "
        f"--target {args.target}",
        "步骤 3/3: 犯罪热点预测 + 热力图数据生成"
    )
    if not ok:
        failed_steps.append("crime_prediction")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    total_time = time.time() - total_t0
    print(f"\n{'='*72}")
    print(f"  Pipeline 完成！")
    print(f"  总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*72}")

    if failed_steps:
        print(f"\n  ⚠  以下步骤执行失败: {failed_steps}")
        print("     请检查错误信息并单独重新运行对应脚本。")

    # 输出文件清单
    expected = [
        ('processed_crime_data.csv',       '预处理后数据（含真实月份 + 时间特征）'),
        ('featured_crime_data.csv',        '特征工程后数据（含空间密度 + total_crimes）'),
        ('tuned_xgboost_model.pkl',        'XGBoost 贝叶斯调优模型'),
        ('tuned_random_forest_model.pkl',  'Random Forest 贝叶斯调优模型'),
        ('core_feature_names.pkl',         '核心特征名称列表'),
        ('scaler.pkl',                     'StandardScaler'),
        ('crime_predictions.csv',          '网格级犯罪预测结果'),
        ('heatmap_data.csv',               '热力图经纬度数据'),
        ('model_comparison.png',           '7种模型 R²/RMSE 对比图'),
        ('shap_summary_xgboost.png',       'SHAP 蜂群图（XGBoost）'),
        ('shap_bar_xgboost.png',           'SHAP 特征重要性图（XGBoost）'),
        ('xgboost_feature_importance.png', 'XGBoost 特征重要性'),
        ('random_forest_feature_importance.png', 'RF 特征重要性'),
        ('monthly_crime_trend.png',        '真实月度犯罪趋势图'),
        ('outcome_distribution.png',       'Last Outcome Category 分布图'),
    ]

    print("\n  生成文件清单:")
    for fname, desc in expected:
        status = "✓" if os.path.exists(fname) else "✗ missing"
        print(f"  [{status:8s}] {fname:42s}  {desc}")

    print(f"\n  启动可视化 Dashboard:")
    print(f"      streamlit run visualization_app.py")
    print(f"\n  完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
