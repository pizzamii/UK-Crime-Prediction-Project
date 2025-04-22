import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime

def load_data(file_path):
    """
    加载CSV数据文件
    """
    print(f"加载数据文件: {file_path}")
    return pd.read_csv(file_path)

def basic_data_exploration(df):
    """
    基本数据探索分析
    """
    print("\n基本数据信息:")
    print(f"数据形状: {df.shape}")
    print("\n数据类型:")
    print(df.dtypes)
    
    print("\n前5行数据:")
    print(df.head())
    
    print("\n基本统计描述:")
    print(df.describe())
    
    print("\n缺失值统计:")
    print(df.isnull().sum())
    
    print("\n唯一值统计:")
    for col in df.columns:
        if df[col].dtype == 'object':
            print(f"{col}: {df[col].nunique()} 个唯一值")

def preprocess_data(df):
    """
    数据预处理
    """
    print("\n开始数据预处理...")
    # 复制数据，避免修改原始数据
    processed_df = df.copy()
    
    # 处理缺失值
    for col in processed_df.columns:
        if processed_df[col].isnull().sum() > 0:
            print(f"处理'{col}'列的缺失值")
            if col == 'crime_id':
                # 对于ID列，可以用UUID或其他唯一标识符填充
                processed_df[col] = processed_df[col].fillna(pd.Series([f"generated_id_{i}" for i in range(processed_df[col].isnull().sum())]))
            elif processed_df[col].dtype == 'float':
                # 对于数值列，用中位数填充
                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
            else:
                # 对于类别列，用众数填充
                processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0])
    
    # 异常值处理（针对经纬度）
    # 英国经度范围大约在-8到2之间
    # 英国纬度范围大约在49到61之间
    processed_df = processed_df[(processed_df['longitude'] >= -8) & (processed_df['longitude'] <= 2)]
    processed_df = processed_df[(processed_df['latitude'] >= 49) & (processed_df['latitude'] <= 61)]
    
    # 添加提取特征 - 从LSOA代码提取区域
    if 'lsoa_code' in processed_df.columns:
        processed_df['lsoa_prefix'] = processed_df['lsoa_code'].str[:3]
    
    # 统一字符串格式
    for col in processed_df.select_dtypes(include=['object']).columns:
        processed_df[col] = processed_df[col].str.strip().str.lower()
    
    # 假设需要添加时间特征
    # 如果数据中没有时间信息，这里可能需要扩展数据源
    # 此处仅为示例，如何添加月份特征（假设我们有日期列）
    if 'date' in processed_df.columns:
        processed_df['month'] = pd.to_datetime(processed_df['date']).dt.month
        processed_df['year'] = pd.to_datetime(processed_df['date']).dt.year
        processed_df['day_of_week'] = pd.to_datetime(processed_df['date']).dt.dayofweek
    else:
        print("注意: 数据中没有日期列，无法提取时间特征")
    
    print("数据预处理完成!")
    return processed_df

def create_crime_counts_by_area(df):
    """
    按地区统计犯罪数量
    """
    return df.groupby(['region', 'crime_type']).size().reset_index(name='count')

def create_crime_counts_by_lsoa(df):
    """
    按LSOA区域统计犯罪数量
    """
    return df.groupby(['lsoa_name', 'crime_type']).size().reset_index(name='count')

def save_processed_data(df, output_file):
    """
    保存处理后的数据
    """
    df.to_csv(output_file, index=False)
    print(f"处理后的数据已保存至: {output_file}")

def visualize_crime_distribution(df):
    """
    可视化犯罪类型分布
    """
    plt.figure(figsize=(12, 6))
    crime_counts = df['crime_type'].value_counts()
    sns.barplot(x=crime_counts.index, y=crime_counts.values)
    plt.xticks(rotation=90)
    plt.title('犯罪类型分布')
    plt.tight_layout()
    plt.savefig('crime_distribution.png')
    plt.close()
    
    if 'region' in df.columns:
        plt.figure(figsize=(12, 6))
        region_counts = df['region'].value_counts().head(10)  # 取前10个地区
        sns.barplot(x=region_counts.index, y=region_counts.values)
        plt.xticks(rotation=45)
        plt.title('地区犯罪数量Top 10')
        plt.tight_layout()
        plt.savefig('region_distribution.png')
        plt.close()

def main():
    # 数据文件路径
    data_file = 'cleaned_street_data.csv'
    output_file = 'processed_crime_data.csv'
    
    # 加载数据
    df = load_data(data_file)
    
    # 数据探索
    basic_data_exploration(df)
    
    # 数据预处理
    processed_df = preprocess_data(df)
    
    # 创建统计特征
    area_crime_counts = create_crime_counts_by_area(processed_df)
    lsoa_crime_counts = create_crime_counts_by_lsoa(processed_df)
    
    print("\n按地区统计的犯罪数量:")
    print(area_crime_counts.head())
    
    print("\n按LSOA区域统计的犯罪数量:")
    print(lsoa_crime_counts.head())
    
    # 可视化犯罪分布
    visualize_crime_distribution(processed_df)
    
    # 保存处理后的数据
    save_processed_data(processed_df, output_file)
    
    print("数据预处理与探索分析完成!")

if __name__ == "__main__":
    main() 