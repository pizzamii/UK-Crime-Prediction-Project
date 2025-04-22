import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os
import argparse
import pymysql
from sqlalchemy import create_engine

def load_data(file_path):
    """
    加载处理后的数据
    """
    print(f"加载数据文件: {file_path}")
    return pd.read_csv(file_path)

# 添加一个保存数据到MySQL的函数
def save_to_mysql(df, table_name, db_config=None):
    """
    将DataFrame保存到MySQL数据库
    
    参数:
    - df: 要保存的DataFrame
    - table_name: 表名
    - db_config: 数据库配置，包含host, user, password, database等信息
    
    返回:
    - 成功则返回True，失败则返回False
    """
    if db_config is None:
        # 默认连接参数
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': '123456',
            'database': 'crime_prediction',
            'port': 3306
        }
    
    try:
        # 创建数据库连接字符串
        connection_str = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        # 创建SQLAlchemy引擎
        engine = create_engine(connection_str)
        
        # 保存DataFrame到MySQL
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        
        print(f"数据已成功保存到MySQL数据库表 '{table_name}'")
        return True
    except Exception as e:
        print(f"保存数据到MySQL失败: {e}")
        return False

def add_geocoding_features(df):
    """
    添加基于地理编码的特征
    """
    print("添加地理编码特征...")
    # 创建新的特征：经纬度网格
    # 将经纬度划分为网格，创建一个粗略的位置特征
    grid_size = 0.1  # 网格大小（经纬度单位）
    df['lon_grid'] = (df['longitude'] / grid_size).astype(int)
    df['lat_grid'] = (df['latitude'] / grid_size).astype(int)
    df['geo_grid'] = df['lon_grid'].astype(str) + '_' + df['lat_grid'].astype(str)
    
    # 计算到主要城市中心的距离
    # 这里使用伦敦作为示例（经度约-0.1278，纬度约51.5074）
    london_lon, london_lat = -0.1278, 51.5074
    df['dist_to_london'] = np.sqrt((df['longitude'] - london_lon)**2 + (df['latitude'] - london_lat)**2)
    
    # 如果有其他城市的坐标，可以计算更多的距离特征
    # 例如，曼彻斯特（经度约-2.2426，纬度约53.4808）
    manchester_lon, manchester_lat = -2.2426, 53.4808
    df['dist_to_manchester'] = np.sqrt((df['longitude'] - manchester_lon)**2 + (df['latitude'] - manchester_lat)**2)
    
    return df

def add_socioeconomic_features(df, socioeconomic_data=None):
    """
    添加社会经济特征，只使用已有数据，不生成假数据
    """
    print("处理社会经济特征...")
    
    if socioeconomic_data is not None:
        # 如果提供了社会经济数据，则合并
        df = pd.merge(df, socioeconomic_data, on='lsoa_code', how='left')
        print("已合并外部社会经济数据")
    else:
        # 如果没有外部数据，则检查数据中是否已包含社会经济指标
        socioeconomic_indicators = [
            col for col in df.columns if any(indicator in col.lower() for indicator in 
                                            ['income', 'population', 'density', 'employment', 'unemployment'])
        ]
        
        if socioeconomic_indicators:
            print(f"数据中已包含以下社会经济指标: {socioeconomic_indicators}")
        else:
            print("警告：未提供社会经济数据，且当前数据中未发现相关指标")
            print("跳过社会经济特征添加步骤")
    
    return df

def add_crime_density_features(df):
    """
    添加犯罪密度特征，适应数据中可能不存在crime_type列的情况
    """
    print("处理犯罪密度特征...")
    
    # 检查是否存在必要的列
    if 'geo_grid' not in df.columns:
        print("警告: 数据中缺少'geo_grid'列，无法添加犯罪密度特征")
        return df
    
    # 检查是否存在crime_type列
    if 'crime_type' in df.columns:
        print("根据crime_type列添加犯罪密度特征...")
        # 按地理网格和犯罪类型计算犯罪密度
        crime_counts = df.groupby(['geo_grid', 'crime_type']).size().reset_index(name='crime_count')
        
        # 创建透视表，行为geo_grid，列为crime_type
        crime_pivot = crime_counts.pivot(index='geo_grid', columns='crime_type', values='crime_count').fillna(0)
        
        # 计算每个网格的总犯罪数
        crime_pivot['total_crimes'] = crime_pivot.sum(axis=1)
        
        # 将犯罪密度特征合并回原始数据
        crime_density = crime_pivot.reset_index()
        df = pd.merge(df, crime_density, on='geo_grid', how='left')
        
        print(f"已添加 {len(crime_pivot.columns)} 种犯罪类型的密度特征")
    else:
        print("数据中缺少'crime_type'列")
        
        # 检查是否已经有犯罪类型列
        crime_type_columns = [col for col in df.columns if any(crime in col.lower() for crime in 
                                                              ['theft', 'burglary', 'robbery', 'violence', 'damage', 'drug'])]
        
        if crime_type_columns:
            print(f"数据中已包含犯罪类型列: {crime_type_columns}")
        else:
            print("只计算总体犯罪密度...")
            # 只计算每个网格的总体犯罪密度
            total_crimes = df.groupby('geo_grid').size().reset_index(name='total_crimes')
            df = pd.merge(df, total_crimes, on='geo_grid', how='left')
            print("已添加'total_crimes'列")
    
    return df

def encode_categorical_features(df):
    """
    对分类特征进行编码
    """
    print("对分类特征进行编码...")
    # 确定哪些列是分类特征
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 移除不需要编码的列（如ID列）
    if 'crime_id' in categorical_cols:
        categorical_cols.remove('crime_id')
    
    # 对每个分类特征进行独热编码
    # 注意：对于基数很高的特征（如lsoa_name），可能需要其他编码方法
    for col in categorical_cols:
        # 对于基数较低的特征使用独热编码
        if df[col].nunique() < 10:  # 假设小于10个唯一值的是基数较低的
            # 创建独热编码
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            # 合并到原始数据
            df = pd.concat([df, dummies], axis=1)
            # 移除原始列
            df.drop(col, axis=1, inplace=True)
        else:
            # 对于基数较高的特征，可以考虑目标编码或其他方法
            # 这里简单地使用标签编码
            df[f'{col}_encoded'] = df[col].factorize()[0]
    
    return df

def create_feature_matrix(df, target_col=None):
    """
    创建特征矩阵和目标变量（如果提供），动态适应数据中存在的列
    """
    print("创建特征矩阵...")
    
    # 定义可能不需要作为特征的列
    potential_drops = ['crime_id', 'location', 'lsoa_name', 'lsoa_code', 'crime_type', 'region']
    
    # 动态检查列是否存在
    features_to_drop = [col for col in potential_drops if col in df.columns]
    
    print(f"将排除以下列: {features_to_drop}")
    
    # 如果提供了目标列，也将其从特征中移除
    if target_col is not None:
        if target_col in df.columns:
            features_to_drop.append(target_col)
            y = df[target_col].values
            print(f"使用 '{target_col}' 作为目标变量")
        else:
            print(f"警告: 目标列 '{target_col}' 不在数据中，不提取目标变量")
            y = None
    else:
        y = None
        print("未指定目标变量")
    
    # 创建特征矩阵
    X = df.drop(features_to_drop, axis=1)
    
    # 返回特征名称，方便后续解释模型
    feature_names = X.columns.tolist()
    print(f"特征矩阵包含 {len(feature_names)} 个特征")
    
    return X, y, feature_names

def normalize_features(X):
    """
    标准化数值特征，提高对各种数据的健壮性
    """
    print("标准化特征...")
    
    # 确保X是DataFrame格式
    if not isinstance(X, pd.DataFrame):
        print("警告: 输入不是DataFrame，尝试转换")
        try:
            X = pd.DataFrame(X)
        except:
            print("错误: 无法将输入转换为DataFrame")
            return X, None
    
    # 确定哪些列是数值特征
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if not numeric_cols:
        print("警告: 未找到数值特征，跳过标准化")
        return X, None
    
    print(f"对以下 {len(numeric_cols)} 个数值特征进行标准化")
    
    # 创建一个副本避免修改原始数据
    X_normalized = X.copy()
    
    # 使用StandardScaler标准化数值特征
    scaler = StandardScaler()
    
    try:
        # 检查是否有空值或无穷值
        has_nan = X[numeric_cols].isna().any().any()
        if has_nan:
            print("警告: 发现空值，将使用均值填充")
            # 对每列单独处理，使用均值填充空值
            for col in numeric_cols:
                if X[col].isna().any():
                    mean_val = X[col].mean()
                    X_normalized[col] = X[col].fillna(mean_val)
        
        # 标准化处理
        X_normalized[numeric_cols] = scaler.fit_transform(X_normalized[numeric_cols])
        print("标准化完成")
        
    except Exception as e:
        print(f"标准化过程中出错: {e}")
        print("尝试单列标准化...")
        
        # 重置scaler
        scaler = StandardScaler()
        
        # 逐列进行标准化，避免整体失败
        for col in numeric_cols:
            try:
                # 填充空值
                if X[col].isna().any():
                    X_normalized[col] = X[col].fillna(X[col].mean())
                
                # 跳过方差为零的列
                if X_normalized[col].var() == 0:
                    print(f"列 '{col}' 方差为零，跳过标准化")
                    continue
                
                # 标准化单列
                X_normalized[col] = scaler.fit_transform(X_normalized[[col]])
            except Exception as col_error:
                print(f"列 '{col}' 标准化失败: {col_error}")
    
    return X_normalized, scaler

def save_features(X, y, feature_names, output_dir='./'):
    """
    保存特征矩阵和目标变量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存特征矩阵
    np.save(os.path.join(output_dir, 'X_features.npy'), X.values)
    
    # 如果有目标变量，也保存它
    if y is not None:
        np.save(os.path.join(output_dir, 'y_target.npy'), y)
    
    # 保存特征名称
    with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)
    
    print(f"特征和目标已保存至: {output_dir}")

def main():
    """
    特征工程主函数，支持命令行参数以增加灵活性
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='对犯罪数据进行特征工程')
    parser.add_argument('--input', type=str, default='processed_crime_data.csv', 
                        help='输入数据文件路径 (默认: processed_crime_data.csv)')
    parser.add_argument('--output', type=str, default='featured_crime_data.csv', 
                        help='输出数据文件路径 (默认: featured_crime_data.csv)')
    parser.add_argument('--target', type=str, default=None,
                        help='目标变量列名 (默认: None，不提取目标变量)')
    parser.add_argument('--no-geocoding', action='store_true',
                        help='跳过添加地理编码特征')
    parser.add_argument('--no-socioeconomic', action='store_true',
                        help='跳过添加社会经济特征')
    parser.add_argument('--no-crime-density', action='store_true',
                        help='跳过添加犯罪密度特征')
    parser.add_argument('--no-categorical-encoding', action='store_true',
                        help='跳过分类特征编码')
    parser.add_argument('--save-to-mysql', action='store_true',
                        help='是否将数据保存到MySQL数据库')
    parser.add_argument('--mysql-table', type=str, default='featured_crime_data',
                        help='MySQL表名 (默认: featured_crime_data)')
    parser.add_argument('--mysql-host', type=str, default='localhost',
                        help='MySQL主机 (默认: localhost)')
    parser.add_argument('--mysql-port', type=int, default=3306,
                        help='MySQL端口 (默认: 3306)')
    parser.add_argument('--mysql-user', type=str, default='root',
                        help='MySQL用户名 (默认: root)')
    parser.add_argument('--mysql-password', type=str, default='',
                        help='MySQL密码 (默认: 空)')
    parser.add_argument('--mysql-database', type=str, default='crime_prediction',
                        help='MySQL数据库名 (默认: crime_prediction)')
    
    args = parser.parse_args()
    
    print("="*80)
    print(f"开始特征工程处理，输入文件: {args.input}")
    print("="*80)
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在")
        return
    
    # 加载处理后的数据
    try:
        df = load_data(args.input)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return
    
    # 添加地理编码特征
    if not args.no_geocoding:
        print("\n步骤1: 添加地理编码特征")
        df = add_geocoding_features(df)
    else:
        print("\n跳过: 添加地理编码特征")
    
    # 添加社会经济特征
    if not args.no_socioeconomic:
        print("\n步骤2: 添加社会经济特征")
        df = add_socioeconomic_features(df)
    else:
        print("\n跳过: 添加社会经济特征")
    
    # 添加犯罪密度特征
    if not args.no_crime_density:
        print("\n步骤3: 添加犯罪密度特征")
        df = add_crime_density_features(df)
    else:
        print("\n跳过: 添加犯罪密度特征")
    
    # 对分类特征进行编码
    if not args.no_categorical_encoding:
        print("\n步骤4: 编码分类特征")
        df = encode_categorical_features(df)
    else:
        print("\n跳过: 编码分类特征")
    
    # 保存特征工程后的数据到CSV
    print(f"\n保存特征工程后的数据到 {args.output}")
    df.to_csv(args.output, index=False)
    print(f"特征工程后的数据已保存到CSV: {df.shape[0]}行 x {df.shape[1]}列")
    
    # 保存数据到MySQL（如果指定了选项）
    if args.save_to_mysql:
        print("\n保存数据到MySQL数据库")
        db_config = {
            'host': args.mysql_host,
            'port': args.mysql_port,
            'user': args.mysql_user,
            'password': args.mysql_password,
            'database': args.mysql_database
        }
        save_to_mysql(df, args.mysql_table, db_config)
    
    # 创建特征矩阵
    print("\n创建特征矩阵和标准化")
    X, y, feature_names = create_feature_matrix(df, args.target)
    
    # 标准化特征
    X, scaler = normalize_features(X)
    
    # 保存特征和目标变量
    print("\n保存特征矩阵和相关数据")
    save_features(X, y, feature_names)
    
    # 保存标准化器
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n特征工程完成!")
    print(f"生成了 {len(feature_names)} 个特征")
    if y is not None:
        print(f"目标变量: {args.target}, 样本数: {len(y)}")

if __name__ == "__main__":
    main() 