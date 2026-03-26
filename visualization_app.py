"""
交互式犯罪数据可视化 Dashboard
================================
基于 Streamlit + Folium + Plotly，对约 480 万条英国公开犯罪记录进行
交互式展示，支持：

  - 按地区 / 犯罪类型 / 时间范围筛选的热力图
  - 月度犯罪趋势图与季节性分析
  - 案件结果（Last Outcome Category）分布
  - 犯罪热点预测排名与预测热力图
  - SHAP 特征重要性可视化

启动命令：
    streamlit run visualization_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ── 页面配置 ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UK Crime Analytics",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.4rem; color: #1E3A8A;
    text-align: center; margin-bottom: 0.8rem;
}
.sub-header {
    font-size: 1.4rem; color: #1E3A8A; margin-bottom: 0.8rem;
}
.info-box {
    background: #EFF6FF; border-left: 5px solid #3B82F6;
    padding: 1rem; margin-bottom: 1rem; border-radius: 4px;
}
.insight-box {
    background: #F0FDF4; border-left: 5px solid #22C55E;
    padding: 1rem; margin-bottom: 1rem; border-radius: 4px;
}
.warning-box {
    background: #FEF2F2; border-left: 5px solid #EF4444;
    padding: 1rem; margin-bottom: 1rem; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── 数据加载（带缓存）─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_csv(path: str):
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"加载 {path} 失败: {e}")
        return None

@st.cache_resource
def load_model(path: str = 'tuned_xgboost_model.pkl'):
    for p in [path, 'tuned_random_forest_model.pkl']:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    return pickle.load(f), p
            except:
                continue
    return None, None

def load_feature_names():
    for p in ['core_feature_names.pkl', 'feature_names.pkl']:
        if os.path.exists(p):
            try:
                with open(p, 'rb') as f:
                    return pickle.load(f)
            except:
                continue
    return None


# ── 绘图函数 ──────────────────────────────────────────────────────────────────

def fig_crime_type(df):
    ct = df['crime_type'].value_counts().reset_index()
    ct.columns = ['Crime Type', 'Count']
    return px.bar(ct, x='Crime Type', y='Count',
                  title='Crime Type Distribution',
                  color='Count', color_continuous_scale='Viridis',
                  height=480)

def fig_region(df, top_n=15):
    rc = df['region'].value_counts().head(top_n).reset_index()
    rc.columns = ['Region', 'Count']
    return px.bar(rc, x='Count', y='Region', orientation='h',
                  title=f'Top {top_n} Regions by Crime Count',
                  color='Count', color_continuous_scale='Plasma',
                  height=500)

def fig_monthly_trend(df):
    if 'month' not in df.columns:
        return None
    mc = df.groupby('month').size().reset_index(name='Count').sort_values('month')
    return px.line(mc, x='month', y='Count',
                   title='Monthly Crime Trend (Real Data from data.police.uk)',
                   markers=True, height=420,
                   labels={'month': 'Month (YYYY-MM)', 'Count': 'Crime Count'})

def fig_seasonal(df):
    if 'season' not in df.columns:
        return None
    season_labels = {1: 'Spring', 2: 'Summer', 3: 'Autumn', 4: 'Winter'}
    sc = df.groupby('season').size().reset_index(name='Count')
    sc['Season'] = sc['season'].map(season_labels)
    colors = {'Spring': '#2ecc71', 'Summer': '#e74c3c',
              'Autumn': '#f39c12', 'Winter': '#3498db'}
    fig = px.bar(sc, x='Season', y='Count',
                  title='Crime by Season', color='Season',
                  color_discrete_map=colors, height=400)
    fig.update_layout(showlegend=False)
    return fig

def fig_yoy(df):
    if 'year' not in df.columns or 'month_num' not in df.columns:
        return None
    yoy = df.groupby(['year', 'month_num']).size().reset_index(name='Count')
    f = px.line(yoy, x='month_num', y='Count', color='year',
                title='Year-over-Year Monthly Comparison', markers=True, height=420)
    f.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan','Feb','Mar','Apr','May','Jun',
                      'Jul','Aug','Sep','Oct','Nov','Dec']
        )
    )
    return f

def fig_holiday_impact(df):
    if 'is_holiday_month' not in df.columns:
        return None
    hc = df.groupby('is_holiday_month').size().reset_index(name='Total')
    nm = df[df['is_holiday_month'] == 0]['month'].nunique() if 'month' in df.columns else 1
    hm = df[df['is_holiday_month'] == 1]['month'].nunique() if 'month' in df.columns else 1
    hc['Avg Monthly'] = hc.apply(
        lambda r: r['Total'] / (hm if r['is_holiday_month'] else nm), axis=1
    )
    hc['Label'] = hc['is_holiday_month'].map({0: 'Non-Holiday', 1: 'Bank Holiday Month'})
    fig = px.bar(hc, x='Label', y='Avg Monthly',
                  title='Avg Monthly Crime: Holiday vs Non-Holiday',
                  color='Label',
                  color_discrete_map={'Non-Holiday': '#3498db',
                                      'Bank Holiday Month': '#e74c3c'},
                  height=380)
    fig.update_layout(showlegend=False)
    return fig

def fig_crime_by_type_monthly(df, top_n=5):
    if 'month' not in df.columns or 'crime_type' not in df.columns:
        return None
    top_types = df['crime_type'].value_counts().head(top_n).index.tolist()
    sub = df[df['crime_type'].isin(top_types)]
    mt = sub.groupby(['month', 'crime_type']).size().reset_index(name='Count')
    mt = mt.sort_values('month')
    return px.line(mt, x='month', y='Count', color='crime_type',
                   title=f'Monthly Trend: Top {top_n} Crime Types',
                   markers=True, height=480)

def fig_outcome(df, top_n=12):
    if 'last_outcome_category' not in df.columns:
        return None
    oc = df['last_outcome_category'].value_counts().head(top_n).reset_index()
    oc.columns = ['Outcome', 'Count']
    return px.bar(oc, x='Count', y='Outcome', orientation='h',
                  title=f'Top {top_n} Last Outcome Categories',
                  color='Count', color_continuous_scale='Purples',
                  height=480)

def fig_outcome_by_crime(df, top_n_crimes=5, top_n_outcomes=5):
    if 'crime_type' not in df.columns or 'last_outcome_category' not in df.columns:
        return None
    top_c = df['crime_type'].value_counts().head(top_n_crimes).index.tolist()
    top_o = df['last_outcome_category'].value_counts().head(top_n_outcomes).index.tolist()
    sub = df[df['crime_type'].isin(top_c) & df['last_outcome_category'].isin(top_o)]
    heat = sub.groupby(['crime_type', 'last_outcome_category']).size().reset_index(name='Count')
    pivot = heat.pivot(index='crime_type', columns='last_outcome_category', values='Count').fillna(0)
    return px.imshow(pivot, title='Crime Type vs Outcome Heatmap',
                     color_continuous_scale='YlOrRd', height=420,
                     labels=dict(color='Count'))

def map_crime_heatmap(df, crime_filter=None, region_filter=None):
    m = folium.Map(location=[54.0, -2.5], zoom_start=6)
    sub = df.copy()
    if crime_filter and crime_filter != 'All Types':
        sub = sub[sub['crime_type'] == crime_filter]
    if region_filter and region_filter != 'All Regions':
        sub = sub[sub['region'] == region_filter]
    if len(sub) > 60000:
        sub = sub.sample(60000, random_state=42)
    heat = [[r['latitude'], r['longitude']] for _, r in sub.iterrows()
            if not (np.isnan(r['latitude']) or np.isnan(r['longitude']))]
    if heat:
        HeatMap(heat, radius=8, blur=10, max_zoom=11).add_to(m)
    return m, len(sub)

def map_prediction_heatmap(hm_data: pd.DataFrame):
    m = folium.Map(location=[54.0, -2.5], zoom_start=6)
    heat = [[r['latitude'], r['longitude'], r['predicted_crimes']]
            for _, r in hm_data.iterrows()]
    if heat:
        HeatMap(heat, radius=10, blur=12, max_zoom=11).add_to(m)
    # Top-10 标记
    top10 = hm_data.sort_values('predicted_crimes', ascending=False).head(10)
    cl = MarkerCluster().add_to(m)
    for _, r in top10.iterrows():
        folium.Marker(
            [r['latitude'], r['longitude']],
            popup=f"Grid: {r['Region']}<br>Predicted: {r['predicted_crimes']:.0f}",
            icon=folium.Icon(color='red', icon='exclamation-sign', prefix='glyphicon')
        ).add_to(cl)
    return m

def fig_top_predictions(pred_df: pd.DataFrame, top_n=20):
    top = pred_df.head(top_n).copy()
    top['Rank'] = range(1, len(top) + 1)
    f = px.bar(top, x='Rank', y='Predicted_Crime',
               title=f'Top {top_n} Predicted Crime Hotspot Grid Cells',
               color='Predicted_Crime', color_continuous_scale='YlOrRd',
               hover_data=['Region'], height=500)
    if 'Lower_Bound' in top.columns and 'Upper_Bound' in top.columns:
        f.add_trace(go.Scatter(x=top['Rank'], y=top['Upper_Bound'],
                               mode='lines', line=dict(width=0), showlegend=False))
        f.add_trace(go.Scatter(x=top['Rank'], y=top['Lower_Bound'],
                               mode='lines', line=dict(width=0),
                               fill='tonexty', fillcolor='rgba(68,68,68,0.18)',
                               showlegend=False))
    return f

def fig_feature_importance(model, feature_names, top_n=20):
    if model is None or not hasattr(model, 'feature_importances_'):
        return None
    imp = model.feature_importances_
    fn  = feature_names if feature_names and len(feature_names) == len(imp) \
        else [f'F{i}' for i in range(len(imp))]
    fi = pd.DataFrame({'Feature': fn, 'Importance': imp}
                      ).sort_values('Importance', ascending=False).head(top_n)
    return px.bar(fi, x='Importance', y='Feature', orientation='h',
                  title=f'Top {top_n} Feature Importances',
                  color='Importance', color_continuous_scale='Viridis',
                  height=520)


# ── 主应用 ────────────────────────────────────────────────────────────────────

def main():
    st.markdown("<h1 class='main-header'>🇬🇧 UK Crime Analytics Dashboard</h1>",
                unsafe_allow_html=True)

    # ── 侧边栏 ──
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Page",
        ["🏠 Home", "🔍 Data Exploration", "📊 Crime Patterns",
         "🕒 Temporal Analysis", "📋 Outcomes Analysis",
         "🔮 Predictions", "ℹ️ About"]
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Source")
    data_file = st.sidebar.selectbox(
        "Select data file",
        ["featured_crime_data.csv", "processed_crime_data.csv", "raw_crime_data.csv"],
        index=0
    )

    # 加载数据
    df          = load_csv(data_file)
    pred_df     = load_csv('crime_predictions.csv')
    heatmap_df  = load_csv('heatmap_data.csv')
    model, mpath = load_model()
    feat_names   = load_feature_names()

    if df is None:
        st.error("数据文件不存在，请先运行完整 pipeline（run_all.py）后再启动 Dashboard。")
        st.code("python run_all.py", language="bash")
        return

    # ══════════════════════════════════════════════════════════════
    # HOME
    # ══════════════════════════════════════════════════════════════
    if page == "🏠 Home":
        st.markdown("<h2 class='sub-header'>Welcome to the UK Crime Analysis Dashboard</h2>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        This dashboard analyses <strong>~4.8 million UK street crime records (May 2024 – Apr 2025)</strong>
        sourced from <a href="https://data.police.uk" target="_blank">data.police.uk</a>,
        featuring real temporal data, spatiotemporal feature engineering, and
        multi-model machine learning with SHAP interpretability.
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", f"{len(df):,}")
        c2.metric("Crime Types", df['crime_type'].nunique() if 'crime_type' in df.columns else 'N/A')
        c3.metric("Regions", df['region'].nunique() if 'region' in df.columns else 'N/A')
        if 'month' in df.columns:
            c4.metric("Time Span", f"{df['month'].min()} ~ {df['month'].max()}")

        if 'crime_type' in df.columns:
            st.plotly_chart(fig_crime_type(df), use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # DATA EXPLORATION
    # ══════════════════════════════════════════════════════════════
    elif page == "🔍 Data Exploration":
        st.markdown("<h2 class='sub-header'>Data Exploration</h2>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Rows",    f"{len(df):,}")
        c2.metric("Columns", len(df.columns))
        c3.metric("Missing", df.isnull().sum().sum())

        st.subheader("Data Preview (first 200 rows)")
        st.dataframe(df.head(200), use_container_width=True)

        st.subheader("Column Info")
        ci = pd.DataFrame({
            'Column': df.columns,
            'Type':   df.dtypes.astype(str).values,
            'Non-Null': df.count().values,
            'Null':   df.isna().sum().values,
            'Unique': [df[c].nunique() for c in df.columns],
        })
        st.dataframe(ci, use_container_width=True)

        num_cols = df.select_dtypes(include='number').columns
        if len(num_cols):
            st.subheader("Numeric Statistics")
            st.dataframe(df[num_cols].describe(), use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # CRIME PATTERNS
    # ══════════════════════════════════════════════════════════════
    elif page == "📊 Crime Patterns":
        st.markdown("<h2 class='sub-header'>Crime Patterns</h2>", unsafe_allow_html=True)

        if 'crime_type' in df.columns:
            st.plotly_chart(fig_crime_type(df), use_container_width=True)

        if 'region' in df.columns:
            top_n_reg = st.slider("Top N Regions", 5, 43, 15)
            st.plotly_chart(fig_region(df, top_n_reg), use_container_width=True)

        # 热力图
        st.subheader("Crime Heatmap (Folium)")
        col1, col2 = st.columns(2)
        with col1:
            types_opts = ['All Types'] + sorted(df['crime_type'].unique().tolist()) \
                if 'crime_type' in df.columns else ['All Types']
            sel_type = st.selectbox("Crime Type", types_opts)
        with col2:
            reg_opts = ['All Regions'] + sorted(df['region'].unique().tolist()) \
                if 'region' in df.columns else ['All Regions']
            sel_reg = st.selectbox("Region", reg_opts)

        if all(c in df.columns for c in ['latitude', 'longitude']):
            crime_map, n_shown = map_crime_heatmap(df, sel_type, sel_reg)
            folium_static(crime_map, width=900, height=520)
            st.caption(f"Showing up to 60,000 of {n_shown:,} matching records")
        else:
            st.warning("No coordinate columns found.")

    # ══════════════════════════════════════════════════════════════
    # TEMPORAL ANALYSIS
    # ══════════════════════════════════════════════════════════════
    elif page == "🕒 Temporal Analysis":
        st.markdown("<h2 class='sub-header'>Temporal Analysis</h2>", unsafe_allow_html=True)

        if 'month' not in df.columns:
            st.warning("No 'month' column found. Please run the preprocessing pipeline first.")
            return

        # 时间范围筛选
        all_months = sorted(df['month'].unique())
        c1, c2 = st.columns(2)
        s_m = c1.selectbox("Start Month", all_months, index=0)
        e_m = c2.selectbox("End Month",   all_months, index=len(all_months) - 1)
        tdf = df[(df['month'] >= s_m) & (df['month'] <= e_m)]
        st.caption(f"Showing {len(tdf):,} records | {s_m} → {e_m}")

        # 月度趋势
        st.subheader("Monthly Crime Trend")
        f = fig_monthly_trend(tdf)
        if f:
            st.plotly_chart(f, use_container_width=True)

        # 季节 + 假日
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Seasonal Distribution")
            f = fig_seasonal(tdf)
            if f:
                st.plotly_chart(f, use_container_width=True)
        with c2:
            st.subheader("Holiday Month Impact")
            f = fig_holiday_impact(tdf)
            if f:
                st.plotly_chart(f, use_container_width=True)

        # 分犯罪类型月度趋势
        st.subheader("Monthly Trend by Crime Type")
        n_types = st.slider("Number of crime types", 3, 10, 5)
        f = fig_crime_by_type_monthly(tdf, n_types)
        if f:
            st.plotly_chart(f, use_container_width=True)

        # 年度对比
        st.subheader("Year-over-Year Comparison")
        f = fig_yoy(tdf)
        if f:
            st.plotly_chart(f, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # OUTCOMES ANALYSIS
    # ══════════════════════════════════════════════════════════════
    elif page == "📋 Outcomes Analysis":
        st.markdown("<h2 class='sub-header'>Case Outcomes Analysis</h2>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        The <em>Last Outcome Category</em> field from data.police.uk records the most recent
        resolution status of each reported crime (e.g., "Under investigation",
        "Unable to prosecute suspect", "Awaiting court outcome", etc.).
        </div>
        """, unsafe_allow_html=True)

        if 'last_outcome_category' not in df.columns:
            st.warning("last_outcome_category 列不存在，请使用 processed_crime_data.csv 或 featured_crime_data.csv")
            return

        top_n_oc = st.slider("Top N Outcomes", 5, 20, 12)
        f = fig_outcome(df, top_n_oc)
        if f:
            st.plotly_chart(f, use_container_width=True)

        st.subheader("Crime Type × Outcome Heatmap")
        n_c = st.slider("Number of crime types", 3, 8, 5)
        n_o = st.slider("Number of outcome types", 3, 8, 5)
        f = fig_outcome_by_crime(df, n_c, n_o)
        if f:
            st.plotly_chart(f, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # PREDICTIONS
    # ══════════════════════════════════════════════════════════════
    elif page == "🔮 Predictions":
        st.markdown("<h2 class='sub-header'>Crime Hotspot Predictions</h2>",
                    unsafe_allow_html=True)

        if pred_df is None:
            st.warning("预测文件 crime_predictions.csv 不存在，请先运行 crime_prediction.py")
            return

        top_n = st.slider("Number of hotspots to display", 5, 50, 20)
        st.plotly_chart(fig_top_predictions(pred_df, top_n), use_container_width=True)

        # 预测热力图
        st.subheader("Prediction Heatmap")
        if heatmap_df is not None:
            pred_map = map_prediction_heatmap(heatmap_df)
            folium_static(pred_map, width=900, height=520)
        else:
            st.warning("heatmap_data.csv 不存在")

        # 特征重要性
        st.subheader("Model Feature Importance")
        if model is not None:
            f = fig_feature_importance(model, feat_names)
            if f:
                st.plotly_chart(f, use_container_width=True)
            if mpath:
                st.caption(f"Model loaded from: {mpath}")
        else:
            st.info("模型文件未找到，请先运行 model_training.py")

        # SHAP 图片
        st.subheader("SHAP Analysis")
        shap_imgs = sorted([f for f in os.listdir('.') if f.startswith('shap_') and f.endswith('.png')])
        if shap_imgs:
            cols = st.columns(min(2, len(shap_imgs)))
            for i, sf in enumerate(shap_imgs):
                cols[i % 2].image(sf, caption=sf.replace('_', ' ').replace('.png', '').title(),
                                  use_container_width=True)
        else:
            st.info("SHAP 图片未找到，请先运行 model_training.py")

        # 最高风险区域
        st.subheader("Prediction Insights")
        top_row = pred_df.iloc[0]
        st.markdown(f"""
        <div class='insight-box'>
        <strong>Highest Risk Grid Cell:</strong> {top_row['Region']}<br>
        <strong>Predicted Crime Count:</strong> {top_row['Predicted_Crime']:.0f}<br>
        <strong>95% Confidence Interval:</strong> {top_row.get('Lower_Bound', 'N/A'):.0f} –
        {top_row.get('Upper_Bound', 'N/A'):.0f}
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # ABOUT
    # ══════════════════════════════════════════════════════════════
    elif page == "ℹ️ About":
        st.markdown("<h2 class='sub-header'>About This Project</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-box'>
        <h3>Project Overview</h3>
        <p>End-to-end machine learning pipeline built on ~4.8 million UK public street crime records
        (May 2024 – Apr 2025) from <a href="https://data.police.uk" target="_blank">data.police.uk</a>
        for spatiotemporal crime prediction.</p>

        <h3>Pipeline Steps</h3>
        <ol>
          <li><strong>Data Download</strong>: data_download.py — downloads monthly archives,
              extracts real <em>Month</em> field and <em>Last Outcome Category</em></li>
          <li><strong>Data Preprocessing</strong>: data_preprocessing.py — missing value handling,
              UK boundary filtering, real temporal feature extraction (year, month_num, season,
              month_sin/cos, weekday/weekend counts, bank holiday markers), outcome encoding</li>
          <li><strong>Feature Engineering</strong>: feature_engineering.py — lat/lon grid encoding
              (0.1° ≈ 11 km), distance to 5 city centres, per-grid crime-type spatial density
              (including bicycle theft & anti-social behaviour), monthly temporal aggregation</li>
          <li><strong>Model Training</strong>: model_training.py — 7-model comparison (OLS, Ridge,
              Lasso, Decision Tree, Random Forest, Gradient Boosting, XGBoost) with Bayesian
              hyperparameter optimisation (Optuna) and SHAP interpretability analysis</li>
          <li><strong>Prediction</strong>: crime_prediction.py — crime hotspot prediction with
              Bootstrap 95% confidence intervals</li>
          <li><strong>Visualisation</strong>: visualization_app.py — this Streamlit + Folium
              interactive dashboard</li>
        </ol>

        <h3>Key Results</h3>
        <ul>
          <li>XGBoost (Bayesian-tuned) achieves R² ≈ 0.934, outperforming Random Forest by ~15 pp</li>
          <li>SHAP identifies <em>bicycle theft spatial density</em> and
              <em>anti-social behaviour density</em> as the strongest predictors</li>
        </ul>

        <h3>Data Source</h3>
        <p><a href="https://data.police.uk/data/" target="_blank">data.police.uk</a> —
           UK Police open street-level crime data</p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            **Tech Stack:**
            - Python · Pandas · NumPy
            - Scikit-learn · XGBoost
            - Optuna (Bayesian Optimisation)
            - SHAP (Explainable AI)
            - Streamlit · Plotly · Folium
            """)
        with c2:
            st.markdown("""
            **7 Models Compared:**
            1. OLS (Linear Regression)
            2. Ridge Regression
            3. Lasso Regression
            4. Decision Tree
            5. Random Forest
            6. Gradient Boosting
            7. XGBoost ← best (R²≈0.934)
            """)


if __name__ == '__main__':
    main()
