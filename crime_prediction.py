import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
import traceback

def load_model(model_path):
    """
    加载预训练模型
    """
    print(f"加载模型: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"加载模型出错: {e}")
        return None

def load_scaler(scaler_path):
    """
    加载特征标准化器
    """
    print(f"加载标准化器: {scaler_path}")
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        print(f"错误: {e}")
        return None

def load_feature_names(feature_names_path):
    """
    加载训练模型时使用的特征名称
    """
    print(f"加载特征名称: {feature_names_path}")
    try:
        with open(feature_names_path, 'rb') as f:
            feature_names = pickle.load(f)
        return feature_names
    except Exception as e:
        print(f"加载特征名称出错: {e}")
        return None

def load_data(file_path):
    """
    加载数据
    """
    print(f"加载数据: {file_path}")
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"加载数据出错: {e}")
        return None

def display_available_crime_types(df):
    """
    显示数据中可用的犯罪类型列表，不假设特定列的存在
    """
    # 预设一些通常不是犯罪类型的列名模式
    non_crime_columns = [
        'latitude', 'longitude', 'geo_grid', 'lsoa_code', 'lsoa_name', 
        'crime_id', 'location', 'region', 'lon_grid', 'lat_grid',
        'dist_to_london', 'dist_to_manchester'
    ]
    
    # 找出数值型列并排除明显的非犯罪类型列
    numeric_cols = df.select_dtypes(include=['number']).columns
    potential_crime_cols = [col for col in numeric_cols if all(keyword not in col.lower() for keyword in 
                                                              ['_encoded', 'density', 'income', 'unemployment', 'population'])]
    
    # 从剩余列中查找可能的犯罪类型列（通常犯罪类型列中会包含'crime'或具有较大的正整数值）
    crime_columns = []
    for col in potential_crime_cols:
        if col not in non_crime_columns and col != 'geo_grid_encoded':
            # 如果列名包含"crime"或是"total"，更可能是犯罪类型
            if 'crime' in col.lower() or 'total' in col.lower() or 'theft' in col.lower() or 'violence' in col.lower():
                crime_columns.append(col)
            # 或者检查是否包含较大的正整数值（可能是计数）
            elif df[col].max() > 5 and df[col].dtype in ['int64', 'float64'] and df[col].min() >= 0:
                crime_columns.append(col)
    
    print("\n可用的犯罪类型列表:")
    for i, crime_type in enumerate(crime_columns, 1):
        print(f"{i}. {crime_type}")
    
    return crime_columns

def prepare_prediction_data(df, target_type, model_type="random_forest"):
    """
    准备预测数据，确保使用与训练时相同的特征
    """
    print(f"准备预测数据，目标类型: {target_type}")
    
    # 优先尝试加载核心特征名称
    core_feature_file = "core_feature_names.pkl"
    if os.path.exists(core_feature_file):
        try:
            with open(core_feature_file, 'rb') as f:
                core_features = pickle.load(f)
            print(f"已加载核心特征: {core_features}")
            
            # 找寻区域标识列
            region_id_col = None
            for col in ['geo_grid', 'lsoa_code', 'region']:
                if col in df.columns:
                    region_id_col = col
                    break
            
            if region_id_col is None:
                print("错误: 无法找到区域标识列")
                return None, None
            
            print(f"使用 {region_id_col} 作为区域标识")
            
            # 按区域聚合数据
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            agg_df = df.groupby(region_id_col)[numeric_cols].mean().reset_index()
            
            # 检查核心特征是否都在数据中
            missing_features = [f for f in core_features if f not in agg_df.columns]
            if missing_features:
                print(f"警告: 缺少以下核心特征: {missing_features}")
                print("尝试使用替代特征...")
                # 对于每个缺失的特征，尝试找一个替代特征
                for missing in missing_features:
                    # 如果是经纬度相关，寻找相似的
                    if 'lat' in missing.lower():
                        for col in agg_df.columns:
                            if 'lat' in col.lower() and col not in core_features:
                                core_features[core_features.index(missing)] = col
                                print(f"用 {col} 替代 {missing}")
                                break
                    elif 'lon' in missing.lower():
                        for col in agg_df.columns:
                            if 'lon' in col.lower() and col not in core_features:
                                core_features[core_features.index(missing)] = col
                                print(f"用 {col} 替代 {missing}")
                                break
                    else:
                        # 寻找任意数值列作为替代
                        for col in numeric_cols:
                            if col not in core_features and col != target_type and col != region_id_col:
                                core_features[core_features.index(missing)] = col
                                print(f"用 {col} 替代 {missing}")
                                break
            
            # 再次检查是否所有特征都可用
            missing_features = [f for f in core_features if f not in agg_df.columns]
            if missing_features:
                print(f"错误: 仍有缺失特征: {missing_features}")
                print("将为缺失特征添加0值列")
                for f in missing_features:
                    agg_df[f] = 0
            
            # 创建特征矩阵，仅使用核心特征
            X = agg_df[core_features].copy()
            print(f"使用的特征: {core_features}")
            print(f"特征矩阵大小: {X.shape}")
            
            return X, agg_df[region_id_col]
            
        except Exception as e:
            print(f"加载核心特征出错: {e}")
            print("尝试使用模型特定的特征名称...")
    
    # 如果核心特征加载失败，尝试加载模型特定的特征名称
    feature_names_file = f"{model_type}_feature_names.pkl"
    if not os.path.exists(feature_names_file):
        print(f"警告: 特征名称文件 {feature_names_file} 不存在，尝试使用通用文件")
        feature_names_file = "feature_names.pkl"
        if not os.path.exists(feature_names_file):
            print(f"错误: 特征名称文件 {feature_names_file} 也不存在")
            return None, None
    
    feature_names = load_feature_names(feature_names_file)
    if feature_names is None:
        print("错误: 无法加载特征名称")
        return None, None
    
    # 寻找区域标识字段
    region_id_col = None
    for col in ['geo_grid', 'lsoa_code', 'region']:
        if col in df.columns:
            region_id_col = col
            break
    
    if region_id_col is None:
        print("错误: 无法找到区域标识列")
        return None, None
    
    print(f"使用 {region_id_col} 作为区域标识")
    
    # 按区域分组并聚合
    if target_type in df.columns:
        # 找出所有数值列，用于聚合
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"使用以下列聚合数据: {numeric_cols}")
        
        # 按区域聚合数据
        agg_df = df.groupby(region_id_col)[numeric_cols].mean().reset_index()
    else:
        print(f"警告: 目标列 '{target_type}' 不在数据中")
        # 找出所有数值列，用于聚合
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        print(f"使用以下列聚合数据: {numeric_cols}")
        
        # 按区域聚合数据
        agg_df = df.groupby(region_id_col)[numeric_cols].mean().reset_index()
    
    # 确保所有需要的特征都存在
    missing_features = [f for f in feature_names if f not in agg_df.columns]
    if missing_features:
        print(f"警告: 缺少以下特征: {missing_features}")
        # 对于缺失的特征，添加0值列
        for f in missing_features:
            agg_df[f] = 0
    
    # 创建特征矩阵，只使用训练时的特征
    X = agg_df[feature_names].copy()
    print(f"使用的特征: {feature_names}")
    print(f"特征矩阵大小: {X.shape}")
    
    return X, agg_df[region_id_col]

def normalize_features(X, scaler=None, model_type="random_forest"):
    """
    标准化特征
    """
    # 如果没有提供标准化器，尝试加载
    if scaler is None:
        scaler_file = f"{model_type}_scaler.pkl"
        if not os.path.exists(scaler_file):
            print(f"警告: 标准化器文件 {scaler_file} 不存在，尝试使用通用文件")
            scaler_file = "scaler.pkl"
            if not os.path.exists(scaler_file):
                print(f"警告: 标准化器文件 {scaler_file} 也不存在，使用新的标准化器")
                scaler = StandardScaler()
                X_normalized = scaler.fit_transform(X)
                return X_normalized
        
        scaler = load_scaler(scaler_file)
        if scaler is None:
            print("警告: 无法加载标准化器，使用新的标准化器")
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            return X_normalized
    
    # 使用标准化器
    try:
        X_normalized = scaler.transform(X)
        return X_normalized
    except Exception as e:
        print(f"标准化特征出错: {e}")
        print("尝试不进行标准化...")
        return X

def calculate_confidence_intervals(model, X, predictions, confidence=0.95, n_bootstraps=1000):
    """
    使用bootstrap方法计算预测的置信区间
    """
    print("计算预测的置信区间...")
    
    # 创建一个空数组来存储bootstrap预测
    bootstrap_predictions = np.zeros((X.shape[0], n_bootstraps))
    
    # 对每个bootstrap样本进行预测
    for i in range(n_bootstraps):
        # 从原始数据中有放回抽样
        indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_bootstrap = X[indices]
        
        # 使用原始模型进行预测
        bootstrap_predictions[:, i] = model.predict(X_bootstrap)
    
    # 计算置信区间
    lower_bound = np.percentile(bootstrap_predictions, ((1 - confidence) / 2) * 100, axis=1)
    upper_bound = np.percentile(bootstrap_predictions, (confidence + (1 - confidence) / 2) * 100, axis=1)
    
    return lower_bound, upper_bound

def create_prediction_results(regions, predictions, lower_bound, upper_bound):
    """
    创建预测结果DataFrame
    """
    print("创建预测结果DataFrame...")
    
    results = pd.DataFrame({
        'Region': regions,
        'Predicted_Crime': predictions,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Confidence_Range': upper_bound - lower_bound
    })
    
    # 按预测值降序排序
    results = results.sort_values('Predicted_Crime', ascending=False)
    
    return results

def visualize_predictions(predictions_df, top_n=20, target_type='total_crimes'):
    """
    可视化预测结果
    """
    print(f"可视化预测结果 (top {top_n})...")
    
    # 提取top_n的预测
    top_predictions = predictions_df.head(top_n)
    
    # 设置图形大小
    plt.figure(figsize=(12, 8))
    
    # 创建条形图
    bar_plot = sns.barplot(x='Predicted_Crime', y='Region', data=top_predictions)
    
    # 添加误差条 (置信区间)
    for i, (_, row) in enumerate(top_predictions.iterrows()):
        plt.errorbar(
            x=row['Predicted_Crime'], 
            y=i,
            xerr=[[row['Predicted_Crime'] - row['Lower_Bound']], 
                 [row['Upper_Bound'] - row['Predicted_Crime']]],
            fmt='none', ecolor='gray', capsize=5
        )
    
    # 添加标题和标签
    plt.title(f'预测的Top {top_n} {target_type} 热点区域')
    plt.xlabel('预测的犯罪数量')
    plt.ylabel('区域')
    plt.tight_layout()
    
    # 保存图形
    output_file = f'predicted_{target_type}_top{top_n}.png'
    plt.savefig(output_file)
    plt.close()
    
    print(f"预测可视化已保存至 {output_file}")

def create_heatmap_data(predictions_df, geo_data_df, region_col='geo_grid'):
    """
    创建热力图数据
    """
    print("准备热力图数据...")
    
    if region_col not in geo_data_df.columns:
        print(f"错误: 地理数据中缺少区域列 '{region_col}'")
        return None
    
    # 合并预测结果与地理数据
    merged_data = pd.merge(
        predictions_df, 
        geo_data_df[[region_col, 'latitude', 'longitude']].drop_duplicates(subset=[region_col]),
        left_on='Region',
        right_on=region_col,
        how='inner'
    )
    
    # 检查是否有匹配的数据
    if merged_data.empty:
        print("错误: 无法匹配预测结果与地理数据")
        return None
    
    # 创建热力图数据
    heatmap_data = merged_data[['latitude', 'longitude', 'Predicted_Crime']].copy()
    
    # 保存热力图数据
    heatmap_data.to_csv('heatmap_data.csv', index=False)
    print("热力图数据已保存至 heatmap_data.csv")
    
    return heatmap_data

def export_prediction_results(results_df, output_file='crime_predictions.csv'):
    """
    导出预测结果
    """
    print(f"导出预测结果至 {output_file}...")
    
    results_df.to_csv(output_file, index=False)
    print(f"预测结果已成功导出至 {output_file}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='预测指定区域的犯罪数量')
    parser.add_argument('--input', type=str, default='featured_crime_data.csv', 
                        help='输入数据文件路径 (默认: featured_crime_data.csv)')
    parser.add_argument('--model', type=str, default='tuned_random_forest_model.pkl',
                        help='模型文件路径 (默认: tuned_random_forest_model.pkl)')
    parser.add_argument('--target', type=str, default='total_crimes',
                        help='目标犯罪类型 (默认: total_crimes)')
    parser.add_argument('--output', type=str, default='crime_predictions.csv',
                        help='输出预测结果的文件路径 (默认: crime_predictions.csv)')
    parser.add_argument('--heatmap', type=str, default='heatmap_data.csv',
                        help='输出热图数据的文件路径 (默认: heatmap_data.csv)')
    args = parser.parse_args()
    
    print("\n开始犯罪预测...")
    print("="*80)
    
    # 确定模型类型
    model_type = "random_forest"
    if "xgboost" in args.model.lower():
        model_type = "xgboost"
    
    # 1. 加载模型
    model = load_model(args.model)
    if model is None:
        print("错误: 无法加载模型，退出预测")
        return
    
    # 2. 加载标准化器
    scaler_path = f"{model_type}_scaler.pkl"
    scaler = load_scaler(scaler_path)
    
    # 3. 加载数据
    df = load_data(args.input)
    if df is None:
        print("错误: 无法加载数据，退出预测")
        return
    
    # 显示可用的犯罪类型
    crime_cols = [col for col in df.columns if col in df.select_dtypes(include=['number']).columns]
    crime_types = sorted([col for col in crime_cols if not any(x in col.lower() for x in ['latitude', 'longitude', 'grid', 'dist'])])
    
    print("\n可用的犯罪类型列表:")
    for i, crime_type in enumerate(crime_types, 1):
        print(f"{i}. {crime_type}")
    
    # 4. 准备预测数据
    X, regions = prepare_prediction_data(df, args.target, model_type)
    if X is None:
        print("错误: 准备数据失败，退出预测")
        return
    
    # 5. 标准化特征
    X_norm = normalize_features(X.values, scaler, model_type)
    
    # 6. 进行预测
    try:
        print("进行预测...")
        predictions = model.predict(X_norm)
        
        # 7. 计算置信区间
        try:
            lower_bound, upper_bound = calculate_confidence_intervals(model, X_norm, predictions)
        except Exception as e:
            print(f"计算置信区间出错: {e}")
            lower_bound = predictions * 0.8
            upper_bound = predictions * 1.2
            print("使用简单方法估计置信区间 (预测值的±20%)")
        
        # 8. 创建预测结果DataFrame
        results_df = pd.DataFrame({
            'Region': regions,
            'Predicted_Crime': predictions,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
        
        # 9. 保存预测结果
        results_df.to_csv(args.output, index=False)
        print(f"预测结果已保存至: {args.output}")
        
        # 10. 准备热图数据（如果有经纬度信息）
        if all(col in df.columns for col in ['latitude', 'longitude']):
            print("准备热图数据...")
            
            # 尝试获取每个区域的平均经纬度
            if 'geo_grid' in df.columns:
                geo_coords = df.groupby('geo_grid')[['latitude', 'longitude']].mean().reset_index()
                
                # 合并预测结果与地理坐标
                heatmap_data = results_df.merge(
                    geo_coords,
                    left_on='Region',
                    right_on='geo_grid',
                    how='inner'
                )
                
                # 为可视化准备数据
                heatmap_data.rename(columns={'Predicted_Crime': 'predicted_crimes'}, inplace=True)
                
                # 保存热图数据
                heatmap_data.to_csv(args.heatmap, index=False)
                print(f"热图数据已保存至: {args.heatmap}")
            else:
                print("警告: 找不到geo_grid列，无法准备热图数据")
        
        print("\n预测完成!")
        
    except Exception as e:
        print(f"预测过程中发生错误: {e}")
        traceback.print_exc()
        print("错误: 预测失败，退出预测")
        return

if __name__ == "__main__":
    main() 