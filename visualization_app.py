import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pickle
import os
from datetime import datetime
import warnings
import pymysql
from sqlalchemy import create_engine
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="UK Crime Data Visualization",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #F3F4F6;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF2F2;
        border-left: 5px solid #EF4444;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 5px solid #3B82F6;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 添加数据库连接功能
def load_data_from_mysql(table_name, db_config=None):
    """
    从MySQL数据库中加载数据
    
    参数:
    - table_name: 表名
    - db_config: 数据库配置，包含host, user, password, database等信息
    
    返回:
    - 加载的DataFrame，失败则返回None
    """
    if db_config is None:
        # 默认连接参数
        db_config = {
            'host': 'localhost',
            'user': 'root',
            'password': 'pass1111',
            'database': 'crime_prediction',
            'port': 3306
        }
    
    try:
        # 创建数据库连接字符串
        connection_str = f"mysql+pymysql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        # 创建SQLAlchemy引擎
        engine = create_engine(connection_str)
        
        # 从MySQL加载数据
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql(query, engine)
        
        return df
    except Exception as e:
        st.error(f"从MySQL加载数据失败: {e}")
        return None

# 加载数据函数
@st.cache_data(ttl=3600)
def load_data(file_path):
    if not os.path.exists(file_path):
        st.error(f"找不到文件: {file_path}")
        return None
    
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"加载数据失败: {e}")
        return None

@st.cache_data
def load_predictions(file_path='crime_predictions.csv'):
    """
    加载预测结果
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.warning(f"加载预测数据出错: {str(e)}")
        return None

@st.cache_data
def load_heatmap_data(file_path='heatmap_data.csv'):
    """
    加载热图数据
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.warning(f"加载热图数据出错: {str(e)}")
        return None

@st.cache_resource
def load_model(model_path='tuned_random_forest_model.pkl'):
    """
    加载模型
    """
    if not os.path.exists(model_path):
        st.warning(f"找不到模型文件: {model_path}")
        return None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"加载模型失败: {e}")
        return None

def plot_crime_distribution(df):
    """
    绘制犯罪类型分布
    """
    crime_counts = df['crime_type'].value_counts().reset_index()
    crime_counts.columns = ['Crime Type', 'Count']
    
    fig = px.bar(
        crime_counts,
        x='Crime Type',
        y='Count',
        title='Crime Type Distribution',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Crime Type',
        yaxis_title='Count',
        xaxis={'categoryorder':'total descending'},
        height=500
    )
    
    return fig

def plot_region_distribution(df, top_n=10):
    """
    绘制地区犯罪分布
    """
    region_counts = df['region'].value_counts().head(top_n).reset_index()
    region_counts.columns = ['Region', 'Count']
    
    fig = px.bar(
        region_counts,
        x='Region',
        y='Count',
        title=f'Top {top_n} Regions by Crime Count',
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis_title='Region',
        yaxis_title='Count',
        xaxis={'categoryorder':'total descending'},
        height=500
    )
    
    return fig

def plot_crime_heatmap(df):
    """
    绘制犯罪热图
    """
    # 创建地图
    m = folium.Map(location=[54.7, -3.0], zoom_start=6)
    
    # 添加热图层
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]
    HeatMap(heat_data).add_to(m)
    
    return m

def plot_prediction_map(heatmap_data):
    """
    绘制预测热图
    """
    # 创建地图
    m = folium.Map(location=[54.7, -3.0], zoom_start=6)
    
    # 添加热图层，使用预测的犯罪数量作为权重
    heat_data = [[row['latitude'], row['longitude'], row['predicted_crimes']] 
                 for _, row in heatmap_data.iterrows()]
    HeatMap(heat_data).add_to(m)
    
    # 为前10个犯罪热点添加标记
    top_spots = heatmap_data.sort_values('predicted_crimes', ascending=False).head(10)
    marker_cluster = MarkerCluster().add_to(m)
    
    for _, row in top_spots.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Region: {row['Region']}<br>Predicted Crimes: {row['predicted_crimes']:.2f}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(marker_cluster)
    
    return m

def plot_top_predictions(predictions_df, top_n=20):
    """
    绘制顶部预测结果
    """
    # 只取前N个区域
    plot_df = predictions_df.head(top_n).copy()
    plot_df['rank'] = range(1, len(plot_df) + 1)
    
    # 创建图表
    fig = px.bar(
        plot_df,
        x='rank',
        y='Predicted_Crime',
        title=f'Top {top_n} Areas by Predicted Crime',
        color='Predicted_Crime',
        color_continuous_scale='Viridis',
        hover_data=['Region']
    )
    
    # 添加置信区间（如果有）
    if 'Lower_Bound' in plot_df.columns and 'Upper_Bound' in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df['rank'],
                y=plot_df['Upper_Bound'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df['rank'],
                y=plot_df['Lower_Bound'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.2)',
                showlegend=False
            )
        )
    
    fig.update_layout(
        xaxis_title='Rank',
        yaxis_title='Predicted Crime Count',
        height=500
    )
    
    return fig

def plot_feature_importance(model, feature_names=None, top_n=20):
    """
    绘制特征重要性，显示真实的特征名称
    """
    if model is None or not hasattr(model, 'feature_importances_'):
        return None

    # 获取特征重要性
    importances = model.feature_importances_

    # 默认使用占位符名字
    if feature_names is None or len(feature_names) != len(importances):
        feature_names = [f"Feature {i+1}" for i in range(len(importances))]

    # 创建DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })

    # 排序取前N个
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False).head(top_n)

    # 创建图表
    fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        title=f'Top {top_n} Feature Importance',
        orientation='h',
        color='Importance',
        color_continuous_scale='Viridis'
    )

    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        yaxis={'categoryorder':'total ascending'},
        height=500
    )

    return fig


def main():
    st.markdown("<h1 class='main-header'>UK Crime Data Analysis and Prediction</h1>", unsafe_allow_html=True)
    
    # 侧边栏
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose a page",
        ["Home", "Data Exploration", "Crime Patterns", "Predictions", "About"]
    )
    
    # 侧边栏 - 数据源选择
    st.sidebar.title("设置")
    data_source = st.sidebar.radio(
        "选择数据源",
        ["CSV文件", "MySQL数据库"]
    )
    
    # 根据数据源选择加载数据
    if data_source == "CSV文件":
        # 从CSV加载
        data_file = st.sidebar.selectbox(
            "选择数据文件",
            ["featured_crime_data.csv", "cleaned_street_data.csv", "processed_crime_data.csv"],
            index=0
        )
        
        df = load_data(data_file)
        if df is None:
            st.error("请确保数据文件存在并且格式正确")
            return
    else:
        # 从MySQL加载
        st.sidebar.subheader("MySQL连接设置")
        mysql_host = st.sidebar.text_input("主机", "localhost")
        mysql_port = st.sidebar.number_input("端口", min_value=1, max_value=65535, value=3306)
        mysql_user = st.sidebar.text_input("用户名", "root")
        mysql_password = st.sidebar.text_input("密码", "pass1111", type="password")
        mysql_database = st.sidebar.text_input("数据库名", "crime_prediction")
        mysql_table = st.sidebar.text_input("表名", "featured_crime_data")
        
        # 连接数据库按钮
        if st.sidebar.button("连接数据库"):
            db_config = {
                'host': mysql_host,
                'port': mysql_port,
                'user': mysql_user,
                'password': mysql_password,
                'database': mysql_database
            }
            
            df = load_data_from_mysql(mysql_table, db_config)
            if df is None:
                st.error("无法连接到MySQL数据库或加载数据")
                return
            else:
                st.success(f"成功从MySQL加载了 {len(df)} 条数据记录")
        else:
            st.warning("请点击'连接数据库'按钮加载数据")
            return
    
    predictions_df = load_predictions()
    heatmap_data = load_heatmap_data()
    model = load_model()
    
    # 主页
    if app_mode == "Home":
        st.markdown("<h2 class='sub-header'>Welcome to the UK Crime Analysis Dashboard</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        This interactive application allows you to explore crime patterns across the UK and visualize predictions about future crime trends.
        Use the sidebar to navigate between different sections of the application.
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='sub-header'>Available Features</h3>", unsafe_allow_html=True)
            st.markdown("""
            - **Data Exploration**: View and explore the raw crime data
            - **Crime Patterns**: Analyze trends and patterns in crime data
            - **Predictions**: Visualize crime predictions and hotspots
            - **About**: Information about this project
            """)
        
        with col2:
            st.markdown("<h3 class='sub-header'>Quick Stats</h3>", unsafe_allow_html=True)
            
            if df is not None:
                st.metric("Total Records", f"{len(df):,}")
                st.metric("Crime Types", df['crime_type'].nunique())
                st.metric("Regions", df['region'].nunique() if 'region' in df.columns else "N/A")
            else:
                st.warning("Data not loaded. Please check if the data file exists.")
        
        # 显示一个示例图
        if df is not None:
            st.markdown("<h3 class='sub-header'>Sample Visualization</h3>", unsafe_allow_html=True)
            fig = plot_crime_distribution(df)
            st.plotly_chart(fig, use_container_width=True)
    
    # 数据探索页面
    elif app_mode == "Data Exploration":
        st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)
        
        if df is None:
            st.error("Failed to load data. Please check if the data file exists.")
            return
        
        # 数据概览
        st.markdown("<h3 class='sub-header'>Data Overview</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Crime Types", df['crime_type'].nunique())
        with col3:
            st.metric("Regions", df['region'].nunique() if 'region' in df.columns else "N/A")
        
        # 数据表格
        st.markdown("<h3 class='sub-header'>Data Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)
        
        # 列信息
        st.markdown("<h3 class='sub-header'>Column Information</h3>", unsafe_allow_html=True)
        
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isna().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # 数据统计
        st.markdown("<h3 class='sub-header'>Data Statistics</h3>", unsafe_allow_html=True)
        
        # 只显示数值列的统计
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        else:
            st.write("No numeric columns found.")
    
    # 犯罪模式页面
    elif app_mode == "Crime Patterns":
        st.markdown("<h2 class='sub-header'>Crime Patterns</h2>", unsafe_allow_html=True)
        
        if df is None:
            st.error("Failed to load data. Please check if the data file exists.")
            return
        
        # 显示犯罪类型分布
        st.markdown("<h3 class='sub-header'>Crime Type Distribution</h3>", unsafe_allow_html=True)
        crime_fig = plot_crime_distribution(df)
        st.plotly_chart(crime_fig, use_container_width=True)
        
        # 显示地区犯罪分布
        st.markdown("<h3 class='sub-header'>Regional Crime Distribution</h3>", unsafe_allow_html=True)
        # 用户选择要显示的顶部地区数量
        top_n = st.slider("Select number of top regions to display", 5, 30, 10)
        region_fig = plot_region_distribution(df, top_n)
        st.plotly_chart(region_fig, use_container_width=True)
        
        # 犯罪热图
        st.markdown("<h3 class='sub-header'>Crime Heatmap</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        This heatmap shows the geographical distribution of crimes. Red areas indicate higher crime density.
        </div>
        """, unsafe_allow_html=True)
        
        # 用户选择要显示的犯罪类型
        crime_types = sorted(df['crime_type'].unique())
        selected_crime_type = st.selectbox("Select crime type to display", 
                                         ["All Types"] + list(crime_types))
        
        # 根据选择过滤数据
        if selected_crime_type == "All Types":
            heatmap_df = df
        else:
            heatmap_df = df[df['crime_type'] == selected_crime_type]
        
        # 显示热图
        crime_map = plot_crime_heatmap(heatmap_df)
        folium_static(crime_map, width=800, height=500)
        
        # 显示洞察
        st.markdown("<h3 class='sub-header'>Key Insights</h3>", unsafe_allow_html=True)
        
        # 计算一些基本洞察
        top_crime = df['crime_type'].value_counts().index[0]
        top_crime_percent = df['crime_type'].value_counts().iloc[0] / len(df) * 100
        
        st.markdown(f"""
        <div class='insight-box'>
        <p><strong>Most Common Crime Type:</strong> {top_crime} ({top_crime_percent:.1f}% of all crimes)</p>
        </div>
        """, unsafe_allow_html=True)
        
        if 'region' in df.columns:
            top_region = df['region'].value_counts().index[0]
            top_region_crime = df[df['region'] == top_region]['crime_type'].value_counts().index[0]
            
            st.markdown(f"""
            <div class='insight-box'>
            <p><strong>Region with Highest Crime Rate:</strong> {top_region}</p>
            <p><strong>Most Common Crime in {top_region}:</strong> {top_region_crime}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 预测页面
    elif app_mode == "Predictions":
        st.markdown("<h2 class='sub-header'>Crime Predictions</h2>", unsafe_allow_html=True)
        
        if predictions_df is None:
            st.warning("Prediction data not found. Please run the prediction script first.")
            if st.button("Show sample predictions instead"):
                # 创建示例预测数据
                sample_regions = [f"Region {i}" for i in range(1, 21)]
                sample_predictions = np.random.normal(100, 30, 20)
                sample_lower = sample_predictions - np.random.uniform(10, 20, 20)
                sample_upper = sample_predictions + np.random.uniform(10, 20, 20)
                
                predictions_df = pd.DataFrame({
                    'region': sample_regions,
                    'predicted_crimes': sample_predictions,
                    'lower_bound': sample_lower,
                    'upper_bound': sample_upper
                })
                
                st.info("Showing sample data for demonstration purposes.")
            else:
                return
        
        # 顶部预测结果
        st.markdown("<h3 class='sub-header'>Top Crime Hotspots</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        These areas are predicted to have the highest crime rates. The darker the color, the higher the predicted crime count.
        </div>
        """, unsafe_allow_html=True)
        
        # 用户选择要显示的顶部区域数量
        top_n = st.slider("Select number of top hotspots to display", 5, 30, 20)
        
        # 显示预测条形图
        pred_fig = plot_top_predictions(predictions_df, top_n)
        st.plotly_chart(pred_fig, use_container_width=True)
        
        # 预测热图
        st.markdown("<h3 class='sub-header'>Prediction Heatmap</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        This heatmap shows the geographical distribution of predicted crimes. Red areas indicate predicted crime hotspots.
        </div>
        """, unsafe_allow_html=True)
        
        if heatmap_data is not None:
            pred_map = plot_prediction_map(heatmap_data)
            folium_static(pred_map, width=800, height=500)
        else:
            st.warning("Heatmap data not found. Unable to display prediction map.")
        
        # 特征重要性
        st.markdown("<h3 class='sub-header'>Feature Importance</h3>", unsafe_allow_html=True)
        
        if model is not None and hasattr(model, 'feature_importances_'):
            # 获取特征名称
            try:
                with open('feature_names.pkl', 'rb') as f:
                    feature_names = pickle.load(f)
            except:
                feature_names = None
            
            # 显示特征重要性图
            importance_fig = plot_feature_importance(model, feature_names)
            st.plotly_chart(importance_fig, use_container_width=True)
            
            st.markdown("""
            <div class='insight-box'>
            <p>The graph above shows which features have the most impact on crime prediction. Features with higher importance scores have more influence on the model's predictions.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Model not found or does not support feature importance analysis.")
        
        # 预测洞察
        st.markdown("<h3 class='sub-header'>Prediction Insights</h3>", unsafe_allow_html=True)
        
        # 计算一些基本洞察
        if predictions_df is not None:
            top_region = predictions_df.iloc[0]['Region']
            top_prediction = predictions_df.iloc[0]['Predicted_Crime']
            
            # 如果有置信区间，也显示它们
            if 'Lower_Bound' in predictions_df.columns and 'Upper_Bound' in predictions_df.columns:
                lower = predictions_df.iloc[0]['Lower_Bound']
                upper = predictions_df.iloc[0]['Upper_Bound']
                
                st.markdown(f"""
                <div class='insight-box'>
                <p><strong>Top Crime Hotspot:</strong> {top_region}</p>
                <p><strong>Predicted Crime Count:</strong> {top_prediction:.2f} (95% CI: {lower:.2f} - {upper:.2f})</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='insight-box'>
                <p><strong>Top Crime Hotspot:</strong> {top_region}</p>
                <p><strong>Predicted Crime Count:</strong> {top_prediction:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # 关于页面
    elif app_mode == "About":
        st.markdown("<h2 class='sub-header'>About This Project</h2>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <h3>Project Overview</h3>
        <p>This project focuses on building a machine learning model to predict crime rates and trends across the UK using historical crime data. The model analyzes past crime patterns, identifying correlations between crime rates and various factors such as location, time, and socio-economic indicators.</p>
        
        <h3>Data Sources</h3>
        <p>The data used in this project comes from publicly available UK crime data sources, processed and enhanced with additional features to improve prediction accuracy.</p>
        
        <h3>Methodology</h3>
        <p>The project follows these steps:</p>
        <ol>
            <li>Data Collection and Preprocessing: Gathering and cleaning historical UK crime data.</li>
            <li>Feature Engineering: Handling missing values, outliers, and enhancing the dataset with additional features.</li>
            <li>Model Development: Developing a machine learning model to predict crime rates and trends based on the processed data.</li>
            <li>Visualization Tool Development: Creating this interactive data visualization tool to represent crime trends over time and geographic regions.</li>
        </ol>
        
        <h3>Key Features</h3>
        <p>The visualization tool provides:</p>
        <ul>
            <li>Interactive maps showing crime hotspots</li>
            <li>Analysis of crime patterns by region and type</li>
            <li>Predictions of future crime trends</li>
            <li>Insights into factors that contribute to crime rates</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='sub-header'>Technical Details</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Technologies Used:**
            - Python
            - Pandas & NumPy for data processing
            - Scikit-learn for machine learning
            - Streamlit for web application
            - Plotly and Folium for visualization
            """)
        
        with col2:
            st.markdown("""
            **Models Explored:**
            - Linear Regression
            - Random Forest
            - Gradient Boosting
            - XGBoost
            """)
        
        st.markdown("<h3 class='sub-header'>Future Improvements</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
        <p>Future enhancements to this project may include:</p>
        <ul>
            <li>Integration with real-time crime data feeds</li>
            <li>More sophisticated time-series analysis</li>
            <li>Addition of socio-economic indicators and other external data sources</li>
            <li>Improved prediction accuracy through model refinement</li>
            <li>Expanded geographic coverage</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 