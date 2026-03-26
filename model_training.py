"""
模型训练模块
============
在 featured_crime_data.csv 上训练并对比 7 种回归模型，
对最优模型进行贝叶斯超参数优化，并输出 SHAP 特征贡献分析。

7 种模型：
  1. OLS (Linear Regression)
  2. Ridge Regression
  3. Lasso Regression
  4. Decision Tree
  5. Random Forest
  6. Gradient Boosting
  7. XGBoost

流程：
  1. 加载特征数据（featured_crime_data.csv）
  2. 按 geo_grid 聚合 → 每行=一个网格，目标变量=total_crimes
  3. 选择核心特征（最多 max_features 个，优先时空+密度特征）
  4. 标准化 + 划分训练/测试集
  5. 训练 7 种基线模型并评估（R², RMSE, MAE）
  6. 贝叶斯超参数优化（Optuna）→ XGBoost & Random Forest
  7. SHAP 分析（蜂群图 + 特征重要性条形图）
  8. 保存模型文件

输入:  featured_crime_data.csv
输出:  tuned_xgboost_model.pkl, tuned_random_forest_model.pkl,
       core_feature_names.pkl, scaler.pkl,
       model_comparison.png, shap_*.png, *_feature_importance.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import shap
import pickle
import os
import time
import warnings
import argparse
import traceback

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')


# ── 数据加载与准备 ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在，请先运行 feature_engineering.py")
    df = pd.read_csv(path)
    print(f"数据加载: {df.shape[0]:,} 行  x  {df.shape[1]} 列")
    return df


def prepare_feature_target(df: pd.DataFrame, target_col: str = 'total_crimes'):
    """
    按 geo_grid 聚合：每个网格一行。
    数值列取均值，target_col 取总和。
    """
    print(f"\n按 geo_grid 聚合，目标变量: {target_col}")

    if 'geo_grid' not in df.columns:
        raise ValueError("缺少 geo_grid 列，请先运行 feature_engineering.py")
    if target_col not in df.columns:
        raise ValueError(f"目标列 '{target_col}' 不存在。可用列: {list(df.columns)[:10]}...")

    num_cols = df.select_dtypes(include='number').columns.tolist()
    agg_dict = {c: ('sum' if c == target_col else 'mean') for c in num_cols if c != 'geo_grid'}

    agg = df.groupby('geo_grid').agg(agg_dict).reset_index()
    feat_cols = [c for c in agg.columns if c not in ('geo_grid', target_col)]
    X = agg[feat_cols]
    y = agg[target_col]

    print(f"  样本数: {len(agg):,}  特征数: {len(feat_cols)}")
    return X, y, feat_cols


def select_core_features(X: pd.DataFrame, feature_names: list, max_features: int = 20):
    """
    按优先级选取核心特征，优先保留时空特征和犯罪密度特征。
    max_features: 最多选取数量
    """
    print(f"\n选择核心特征（最多 {max_features} 个）...")

    # 优先级关键词（按顺序）
    PRIORITY = [
        # 城市距离（空间位置）
        'dist_to_london', 'dist_to_manchester', 'dist_to_birmingham',
        'dist_to_leeds', 'dist_to_glasgow',
        # 网格坐标
        'lon_grid', 'lat_grid', 'longitude', 'latitude',
        # 犯罪密度（SHAP 最重要特征）
        'bicycle_theft', 'anti_social_behaviour',
        'violence_and_sexual_offences', 'vehicle_crime',
        'burglary', 'shoplifting', 'public_order',
        'other_theft', 'criminal_damage_and_arson',
        # 时间聚合
        'monthly_crime_mean', 'monthly_crime_std', 'monthly_crime_range',
        'monthly_crime_max',
        # 时间特征
        'month_sin', 'month_cos', 'month_num', 'year', 'season',
        'n_weekdays', 'n_weekends', 'n_bank_holidays', 'is_holiday_month',
        # 结果编码
        'last_outcome_encoded',
    ]

    selected = []
    for kw in PRIORITY:
        for f in feature_names:
            if kw.lower() in f.lower() and f not in selected:
                selected.append(f)
                if len(selected) >= max_features:
                    break
        if len(selected) >= max_features:
            break

    # 补充剩余特征
    if len(selected) < max_features:
        for f in feature_names:
            if f not in selected:
                selected.append(f)
                if len(selected) >= max_features:
                    break

    # 确保所有特征都在 X 中
    selected = [f for f in selected if f in X.columns]
    print(f"  已选 {len(selected)} 个核心特征: {selected}")
    return selected


# ── 模型训练与评估 ─────────────────────────────────────────────────────────────

def train_all_models(X_train, y_train, random_state: int = 42) -> dict:
    """训练 7 种回归模型"""
    print("\n训练 7 种基线模型...")
    print("-" * 60)

    models = {
        'OLS (Linear Regression)': LinearRegression(),
        'Ridge Regression':        Ridge(alpha=1.0),
        'Lasso Regression':        Lasso(alpha=0.1, max_iter=5000),
        'Decision Tree':           DecisionTreeRegressor(random_state=random_state, max_depth=10),
        'Random Forest':           RandomForestRegressor(n_estimators=100, n_jobs=-1,
                                                         random_state=random_state),
        'Gradient Boosting':       GradientBoostingRegressor(n_estimators=100,
                                                              random_state=random_state),
        'XGBoost':                 XGBRegressor(n_estimators=100, verbosity=0,
                                                n_jobs=-1, random_state=random_state),
    }

    trained = {}
    for name, m in models.items():
        t0 = time.time()
        m.fit(X_train, y_train)
        trained[name] = {'model': m, 'time': time.time() - t0}
        print(f"  {name:30s}: {trained[name]['time']:.2f}s")
    return trained


def evaluate_models(trained: dict, X_train, y_train, X_test, y_test) -> dict:
    """评估所有模型"""
    print("\n模型评估结果:")
    print("-" * 80)

    results = {}
    for name, info in trained.items():
        m = info['model']
        metrics = {
            'train_r2':   r2_score(y_train, m.predict(X_train)),
            'test_r2':    r2_score(y_test,  m.predict(X_test)),
            'train_rmse': np.sqrt(mean_squared_error(y_train, m.predict(X_train))),
            'test_rmse':  np.sqrt(mean_squared_error(y_test,  m.predict(X_test))),
            'train_mae':  mean_absolute_error(y_train, m.predict(X_train)),
            'test_mae':   mean_absolute_error(y_test,  m.predict(X_test)),
            'time':       info['time'],
        }
        results[name] = metrics
        print(f"\n  {name}")
        print(f"    训练 R²: {metrics['train_r2']:.4f}  |  测试 R²: {metrics['test_r2']:.4f}")
        print(f"    测试 RMSE: {metrics['test_rmse']:.4f}  |  测试 MAE: {metrics['test_mae']:.4f}")
    return results


def plot_model_comparison(results: dict, output_dir: str = './'):
    """绘制 7 种模型性能对比图"""
    print("\n绘制模型对比图...")

    names = list(results.keys())
    short  = [n.split('(')[0].strip() for n in names]
    r2_vals  = [results[n]['test_r2']   for n in names]
    rmse_vals = [results[n]['test_rmse'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # R² 柱状图
    colors = ['#e74c3c' if v < 0.5 else '#f39c12' if v < 0.8 else '#2ecc71' for v in r2_vals]
    bars = axes[0].bar(short, r2_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_title('Model Comparison — Test R² (Higher is Better)',
                       fontsize=12, fontweight='bold')
    axes[0].set_ylabel('R²')
    axes[0].set_ylim(0, min(1.05, max(r2_vals) * 1.15) if r2_vals else 1)
    for bar, val in zip(bars, r2_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    axes[0].tick_params(axis='x', rotation=35)

    # RMSE 柱状图
    bars2 = axes[1].bar(short, rmse_vals, color='#3498db', edgecolor='black', linewidth=0.5)
    axes[1].set_title('Model Comparison — Test RMSE (Lower is Better)',
                       fontsize=12, fontweight='bold')
    axes[1].set_ylabel('RMSE')
    for bar, val in zip(bars2, rmse_vals):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    axes[1].tick_params(axis='x', rotation=35)

    plt.tight_layout()
    out = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存 {out}")


# ── 贝叶斯超参数优化 ──────────────────────────────────────────────────────────

def _grid_fallback_xgb(X_train, y_train, X_test, y_test, random_state):
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
    }
    grid = GridSearchCV(
        XGBRegressor(random_state=random_state, verbosity=0),
        param_grid, cv=3, scoring='r2', n_jobs=-1
    )
    grid.fit(X_train, y_train)
    m = grid.best_estimator_
    r2 = r2_score(y_test, m.predict(X_test))
    return m, grid.best_params_, r2


def bayesian_optimize_xgboost(X_train, y_train, X_test, y_test,
                               n_trials: int = 50, random_state: int = 42):
    """使用 Optuna 对 XGBoost 进行贝叶斯超参数优化"""
    print(f"\nXGBoost 贝叶斯优化 ({n_trials} trials)...")

    if not HAS_OPTUNA:
        print("  Optuna 未安装，回退到 GridSearchCV...")
        return _grid_fallback_xgb(X_train, y_train, X_test, y_test, random_state)

    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 100, 600),
            'max_depth':        trial.suggest_int('max_depth', 3, 12),
            'learning_rate':    trial.suggest_float('learning_rate', 0.005, 0.3, log=True),
            'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma':            trial.suggest_float('gamma', 0, 5),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }
        m = XGBRegressor(**p, random_state=random_state, verbosity=0, n_jobs=-1)
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_model  = XGBRegressor(**best_params, random_state=random_state, verbosity=0, n_jobs=-1)
    best_model.fit(X_train, y_train)

    y_tr_pred = best_model.predict(X_train)
    y_te_pred = best_model.predict(X_test)
    best_r2   = r2_score(y_test, y_te_pred)

    print(f"  最佳参数: {best_params}")
    print(f"  训练 R²: {r2_score(y_train, y_tr_pred):.4f}  |  测试 R²: {best_r2:.4f}")
    print(f"  测试 RMSE: {np.sqrt(mean_squared_error(y_test, y_te_pred)):.4f}")
    return best_model, best_params, best_r2


def bayesian_optimize_random_forest(X_train, y_train, X_test, y_test,
                                    n_trials: int = 30, random_state: int = 42):
    """使用 Optuna 对 Random Forest 进行贝叶斯超参数优化"""
    print(f"\nRandom Forest 贝叶斯优化 ({n_trials} trials)...")

    if not HAS_OPTUNA:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
        grid = GridSearchCV(RandomForestRegressor(random_state=random_state),
                            param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        m = grid.best_estimator_
        r2 = r2_score(y_test, m.predict(X_test))
        return m, grid.best_params_, r2

    def objective(trial):
        p = {
            'n_estimators':     trial.suggest_int('n_estimators', 50, 400),
            'max_depth':        trial.suggest_int('max_depth', 5, 40),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features':     trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        }
        m = RandomForestRegressor(**p, random_state=random_state, n_jobs=-1)
        m.fit(X_train, y_train)
        return r2_score(y_test, m.predict(X_test))

    study = optuna.create_study(direction='maximize',
                                 sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_model = RandomForestRegressor(**study.best_params, random_state=random_state, n_jobs=-1)
    best_model.fit(X_train, y_train)
    best_r2 = r2_score(y_test, best_model.predict(X_test))

    print(f"  最佳参数: {study.best_params}")
    print(f"  调优后 RF R²: {best_r2:.4f}")
    return best_model, study.best_params, best_r2


# ── SHAP 分析 ─────────────────────────────────────────────────────────────────

def shap_analysis(model, X_test, feature_names: list,
                  model_name: str = 'XGBoost', output_dir: str = './'):
    """SHAP 蜂群图 + 条形图"""
    print(f"\nSHAP 分析 ({model_name})...")

    try:
        X_arr = np.array(X_test) if not isinstance(X_test, np.ndarray) else X_test

        # 采样（大数据集时）
        n = X_arr.shape[0]
        if n > 1000:
            idx   = np.random.choice(n, 1000, replace=False)
            X_smp = X_arr[idx]
        else:
            X_smp = X_arr

        # 创建 DataFrame 以保留特征名
        X_smp_df = pd.DataFrame(X_smp, columns=feature_names)

        # 优先 TreeExplainer，失败则用通用 Explainer
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(X_smp_df, check_additivity=False)
        except Exception as te:
            print(f"  TreeExplainer 失败 ({te})，尝试通用 Explainer...")
            explainer = shap.Explainer(model, X_smp_df)
            shap_vals = explainer(X_smp_df)
            sv = shap_vals.values

        # 如果返回 list（多输出），取第一个
        if isinstance(sv, list):
            sv = sv[0] if len(sv) == 1 else sv[1]

        print(f"  SHAP values shape: {sv.shape}")

        # 蜂群图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(sv, X_smp_df, feature_names=feature_names,
                          show=False, max_display=20)
        plt.title(f'SHAP Feature Importance — {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fname = f'shap_summary_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {fname}")

        # 条形图
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, X_smp_df, feature_names=feature_names,
                          plot_type='bar', show=False, max_display=20)
        plt.title(f'SHAP Mean |Value| — {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        fname2 = f'shap_bar_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(os.path.join(output_dir, fname2), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  已保存: {fname2}")

        # Top-10 输出
        mean_abs = np.abs(sv).mean(axis=0)
        top10 = np.argsort(mean_abs)[::-1][:10]
        print("  SHAP Top-10 特征贡献:")
        for rank, idx in enumerate(top10, 1):
            print(f"    {rank:2d}. {feature_names[idx]:35s}: {mean_abs[idx]:.4f}")

        return sv

    except Exception as e:
        print(f"  SHAP 分析出错: {e}")
        traceback.print_exc()
        return None


def plot_feature_importance(model, feature_names: list, model_name: str, output_dir: str = './'):
    """传统特征重要性图（树模型）"""
    if not hasattr(model, 'feature_importances_'):
        return

    fi = pd.DataFrame({'Feature': feature_names,
                       'Importance': model.feature_importances_}
                      ).sort_values('Importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=fi.head(20), palette='viridis')
    plt.title(f'{model_name} Feature Importance (Top 20)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fname = f'{model_name.lower().replace(" ", "_")}_feature_importance.png'
    plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存 {fname}")
    return fi


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='犯罪预测模型训练（7 种模型 + 贝叶斯优化 + SHAP）')
    parser.add_argument('--input',         type=str,   default='featured_crime_data.csv')
    parser.add_argument('--target',        type=str,   default='total_crimes')
    parser.add_argument('--test_size',     type=float, default=0.2)
    parser.add_argument('--random_state',  type=int,   default=42)
    parser.add_argument('--optuna_trials', type=int,   default=50,
                        help='贝叶斯优化 trial 数（默认 50）')
    parser.add_argument('--max_features',  type=int,   default=20,
                        help='核心特征数量上限（默认 20）')
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"模型训练 | 目标: {args.target}")
    print(f"{'='*70}")

    # 1. 加载
    df = load_data(args.input)

    if args.target not in df.columns:
        print(f"错误: 目标列 '{args.target}' 不存在")
        return

    # 2. 准备特征
    try:
        X, y, feature_names = prepare_feature_target(df, args.target)
        core = select_core_features(X, feature_names, args.max_features)
        X_core = X[core]

        # 保存特征名
        with open('core_feature_names.pkl', 'wb') as f:
            pickle.dump(core, f)
        with open('feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)

        # 标准化
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X_core)
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        X_train, X_test, y_train, y_test = train_test_split(
            X_sc, y, test_size=args.test_size, random_state=args.random_state
        )
        print(f"\n  训练集: {X_train.shape[0]:,}  |  测试集: {X_test.shape[0]:,}")

    except Exception as e:
        print(f"数据准备出错: {e}")
        traceback.print_exc()
        return

    # 3. 7种基线模型
    try:
        trained = train_all_models(X_train, y_train, args.random_state)
        results = evaluate_models(trained, X_train, y_train, X_test, y_test)
        plot_model_comparison(results)
        best_base = max(results, key=lambda x: results[x]['test_r2'])
        print(f"\n基线最佳: {best_base}  (R²={results[best_base]['test_r2']:.4f})")
    except Exception as e:
        print(f"模型训练出错: {e}")
        traceback.print_exc()
        return

    # 4. 贝叶斯优化 — XGBoost
    xgb_r2 = None
    try:
        tuned_xgb, xgb_params, xgb_r2 = bayesian_optimize_xgboost(
            X_train, y_train, X_test, y_test,
            n_trials=args.optuna_trials, random_state=args.random_state
        )
        with open('tuned_xgboost_model.pkl', 'wb') as f:
            pickle.dump(tuned_xgb, f)
        plot_feature_importance(tuned_xgb, core, 'XGBoost')
    except Exception as e:
        print(f"XGBoost 优化出错: {e}")
        traceback.print_exc()
        tuned_xgb = None

    # 5. 贝叶斯优化 — Random Forest
    rf_r2 = None
    try:
        tuned_rf, rf_params, rf_r2 = bayesian_optimize_random_forest(
            X_train, y_train, X_test, y_test,
            n_trials=30, random_state=args.random_state
        )
        with open('tuned_random_forest_model.pkl', 'wb') as f:
            pickle.dump(tuned_rf, f)
        plot_feature_importance(tuned_rf, core, 'Random Forest')
    except Exception as e:
        print(f"RF 优化出错: {e}")
        traceback.print_exc()
        tuned_rf = None

    # 6. SHAP 分析
    if tuned_xgb:
        shap_analysis(tuned_xgb, X_test, core, 'XGBoost')
    if tuned_rf:
        shap_analysis(tuned_rf, X_test, core, 'Random Forest')

    # 7. 汇总
    print(f"\n{'='*70}")
    print("训练完成！结果汇总")
    print(f"{'='*70}")
    print(f"\n7 种基线模型 R²（降序）:")
    for n, m in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
        print(f"  {n:30s}  R²={m['test_r2']:.4f}  RMSE={m['test_rmse']:.2f}")

    print(f"\n贝叶斯优化结果:")
    if xgb_r2 is not None:
        print(f"  XGBoost (tuned)       R² = {xgb_r2:.4f}")
    if rf_r2 is not None:
        print(f"  Random Forest (tuned) R² = {rf_r2:.4f}")
    if xgb_r2 and rf_r2:
        diff = xgb_r2 - rf_r2
        print(f"  XGBoost 比 RF 高 {diff:.4f} ({diff*100:.1f} 个百分点)")

    print(f"\n核心特征 ({len(core)}): {core}")


if __name__ == '__main__':
    main()
