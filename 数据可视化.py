from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, PCA
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.clustering import KMeans
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties, FontManager
from io import BytesIO
import base64
from jinja2 import Template
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# ----------------------
# 全局字体设置
# ----------------------
def set_chinese_font(font_path="/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc"):
    """设置全局中文字体为文泉驿正黑，支持错误处理和字体缓存"""
    try:
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"字体文件未找到：{font_path}")

        font_prop = FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 为seaborn设置字体
        sns.set_style("whitegrid", {"font.family": font_prop.get_name()})

        # 刷新字体缓存
        fm = FontManager()
        fm.addfont(font_path)

        print(f"[SUCCESS] 使用字体：{font_prop.get_name()}，路径：{font_path}")
        return font_prop

    except Exception as e:
        print(f"[ERROR] 字体设置失败：{e}")
        print("[WARNING] 将使用系统默认字体（可能导致中文显示异常）")
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        return None

# 设置字体
font_prop = set_chinese_font()

# 忽略无关警告
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", module="py4j")

# 环境配置
java_home = "/export/server/jdk1.8.0_241/"
os.environ['JAVA_HOME'] = java_home
os.environ['PATH'] = f"{java_home}/bin:{os.environ['PATH']}"

# ----------------------
# 数据读取与清洗
# ----------------------
try:
    df = pd.read_csv('traffic1.csv')
    print("数据基本信息：")
    df.info()

    # 数据质量检查
    print("\n数据质量检查：")
    print(f"重复行数: {df.duplicated().sum()}")
    print(f"缺失值总数: {df.isnull().sum().sum()}")

    # 查看关键列分布
    print("\n事故严重程度分布:")
    print(df['Accident_Severity'].value_counts())

except Exception as e:
    print(f"[ERROR] 读取数据失败: {e}")
    exit()

print("\n开始数据清洗...")
# 缺失值处理
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# 异常值处理（数值型特征）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'id']
for col in numeric_cols:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    df[col] = df[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)

# 时间特征工程
df['day'] = pd.to_datetime(df['day'])
df['month'] = df['day'].dt.month
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')
df['hour'] = df['time'].dt.hour
df['is_peak_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9)) | ((df['hour'] >= 17) & (df['hour'] <= 19))
df['is_peak_hour'] = df['is_peak_hour'].astype(int)

# 删除低方差分类特征
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].nunique() <= 1:
        df = df.drop(col, axis=1)
print("数据清洗完成!\n")

# ----------------------
# Spark 初始化
# ----------------------
spark = SparkSession.builder \
    .appName('TrafficAccidentAnalysis') \
    .config('spark.driver.memory', '4g') \
    .config('spark.executor.memory', '4g') \
    .config('spark.serializer', 'org.apache.spark.serializer.KryoSerializer') \
    .config('spark.sql.shuffle.partitions', 20) \
    .config('spark.default.parallelism', 20) \
    .getOrCreate()

spark_df = spark.createDataFrame(df.reset_index(drop=True))

# ----------------------
# 数据集划分（在特征工程前进行）
# ----------------------
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)
print(f"训练集样本数: {train_data.count()}, 测试集样本数: {test_data.count()}")

# ----------------------
# 特征工程（严格区分训练集和测试集）
# ----------------------
# 字符串编码（仅在训练集拟合）
string_cols = [col for col, dtype in train_data.dtypes if dtype == 'string' and col != 'Accident_Severity']
indexers = {}
for col in string_cols:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_indexed").fit(train_data)
    indexers[col] = indexer
    train_data = indexer.transform(train_data)
    test_data = indexer.transform(test_data)

# 标签编码
label_indexer = StringIndexer(inputCol='Accident_Severity', outputCol='Accident_Severity_indexed').fit(train_data)
train_data = label_indexer.transform(train_data)
test_data = label_indexer.transform(test_data)

# 特征组装（索引特征 + 数值特征）
feature_cols = [f"{col}_indexed" for col in string_cols] + [
    col for col, dtype in train_data.dtypes
    if dtype in ['int', 'double'] and col not in ['id', 'Accident_Severity', 'Accident_Severity_indexed']
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
train_data = assembler.transform(train_data)
test_data = assembler.transform(test_data)

print("\n最终用于建模的特征:")
print(feature_cols)

# ----------------------
# 多分类随机森林模型
# ----------------------
print("\n==== 多分类模型训练 ====")
# 降低模型复杂度，防止过拟合
rf = RandomForestClassifier(
    labelCol='Accident_Severity_indexed',
    featuresCol='features',
    numTrees=30,  # 减少树的数量
    maxDepth=4,  # 降低树的深度
    impurity='gini',
    featureSubsetStrategy='sqrt',  # 每棵树随机选择部分特征
    seed=42
)

# 多维度评估
evaluator_accuracy = MulticlassClassificationEvaluator(
    labelCol='Accident_Severity_indexed',
    metricName='accuracy'
)

evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol='Accident_Severity_indexed',
    metricName='f1'
)

# 训练模型
rf_model = rf.fit(train_data)
predictions_multi = rf_model.transform(test_data)

# 评估模型
accuracy = evaluator_accuracy.evaluate(predictions_multi)
f1_score = evaluator_f1.evaluate(predictions_multi)

print(f"多分类模型准确率: {accuracy:.4f}")
print(f"多分类模型F1分数: {f1_score:.4f}")

# 特征重要性分析
print("\n==== 多分类模型特征重要性 ====")
feature_importance = pd.DataFrame({
    "特征": feature_cols,
    "重要性": rf_model.featureImportances.toArray()
}).sort_values(by="重要性", ascending=False)

print(feature_importance.head(10))

# ----------------------
# 二分类模型（严重事故预测）
# ----------------------
print("\n==== 二分类模型训练 ====")
# 创建二分类标签
train_data_bin = train_data.withColumn(
    "is_severe",
    F.when(F.col("Accident_Severity_indexed") > 0, 1).otherwise(0).cast("integer")
)
test_data_bin = test_data.withColumn(
    "is_severe",
    F.when(F.col("Accident_Severity_indexed") > 0, 1).otherwise(0).cast("integer")
)

# 检查标签分布
print("\n训练集二分类标签分布:")
train_data_bin.groupBy("is_severe").count().show()

# 二分类随机森林模型
rf_bin = RandomForestClassifier(
    labelCol='is_severe',
    featuresCol='features',
    numTrees=30,
    maxDepth=4,
    impurity='gini',
    featureSubsetStrategy='sqrt',
    seed=42
)

# 评估器
evaluator_bin_auc = BinaryClassificationEvaluator(
    labelCol='is_severe',
    metricName='areaUnderROC'
)

evaluator_bin_pr = BinaryClassificationEvaluator(
    labelCol='is_severe',
    metricName='areaUnderPR'
)

# 训练模型
rf_bin_model = rf_bin.fit(train_data_bin)
predictions_bin = rf_bin_model.transform(test_data_bin)

# 评估模型
auc = evaluator_bin_auc.evaluate(predictions_bin)
pr = evaluator_bin_pr.evaluate(predictions_bin)

print(f"二分类模型 AUC: {auc:.4f}")
print(f"二分类模型 PR曲线下面积: {pr:.4f}")

# 二分类特征重要性
print("\n==== 二分类模型特征重要性 ====")
bin_feature_importance = pd.DataFrame({
    "特征": feature_cols,
    "重要性": rf_bin_model.featureImportances.toArray()
}).sort_values(by="重要性", ascending=False)

print(bin_feature_importance.head(10))

# ----------------------
# K-Means聚类分析
# ----------------------
print("\n==== K-Means聚类 ====")
# 标准化特征
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True)
scaler_model = scaler.fit(train_data)
train_scaled = scaler_model.transform(train_data)
test_scaled = scaler_model.transform(test_data)

# 合并数据进行聚类
clustering_data = train_scaled.union(test_scaled).select("scaledFeatures").repartition(10)

# KMeans聚类
kmeans = KMeans(k=3, featuresCol="scaledFeatures", predictionCol="prediction", seed=42)
kmeans_model = kmeans.fit(clustering_data)
clustered_data = kmeans_model.transform(clustering_data)

# 打印聚类分布
print("\n聚类分布:")
clustered_data.groupBy("prediction").count().show()

# PCA降维（应用于聚类后的数据，添加pcaFeatures列）
pca = PCA(k=2, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_model = pca.fit(clustered_data)
clustered_data = pca_model.transform(clustered_data)

print(f"\nPCA方差解释率: {pca_model.explainedVariance}")
print(f"总方差解释率: {sum(pca_model.explainedVariance):.4f}")

# 将PCA特征分解为单独的列以便绘图
clustered_data = clustered_data.withColumn("pca_x", vector_to_array("pcaFeatures")[0])
clustered_data = clustered_data.withColumn("pca_y", vector_to_array("pcaFeatures")[1])

# 转换为Pandas DataFrame进行绘图
pandas_data = clustered_data.select("prediction", "pca_x", "pca_y").toPandas()

# 输出聚类中心（需将NumPy数组转换为Spark向量）
centers_scaled = kmeans_model.clusterCenters()

# 关键修改：将NumPy数组转换为Spark向量
centers_df = spark.createDataFrame([
    {"scaledFeatures": Vectors.dense(center)} for center in centers_scaled
])

centers_pca = pca_model.transform(centers_df)
centers_pca = centers_pca.withColumn("pca_x", vector_to_array("pcaFeatures")[0])
centers_pca = centers_pca.withColumn("pca_y", vector_to_array("pcaFeatures")[1])
centers_pd = centers_pca.select("pca_x", "pca_y").toPandas()

print("\n聚类中心（标准化后）:")
for i, center in enumerate(centers_scaled):
    print(f"聚类{i + 1}: {np.round(center, 2)}")

# ----------------------
# 生成可视化图表
# ----------------------
# 1. 事故严重程度分布饼图
severity_counts = df['Accident_Severity'].value_counts()
fig_pie = px.pie(
    names=severity_counts.index,
    values=severity_counts.values,
    title='事故严重程度分布',
    color_discrete_sequence=px.colors.qualitative.Pastel
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label')
severity_pie_img = fig_pie.to_html(full_html=False)

# 2. 多分类模型特征重要性图
top_features = feature_importance.head(10)
fig_multi = px.bar(
    top_features,
    x='重要性',
    y='特征',
    orientation='h',
    title='多分类模型特征重要性排名',
    color='重要性',
    color_continuous_scale='Viridis'
)
multi_feature_img = fig_multi.to_html(full_html=False)

# 3. 二分类模型特征重要性图
top_bin_features = bin_feature_importance.head(10)
fig_binary = px.bar(
    top_bin_features,
    x='重要性',
    y='特征',
    orientation='h',
    title='二分类模型特征重要性排名',
    color='重要性',
    color_continuous_scale='Magma'
)
binary_feature_img = fig_binary.to_html(full_html=False)

# 4. 聚类分析散点图
fig_cluster = px.scatter(
    pandas_data,
    x='pca_x',
    y='pca_y',
    color='prediction',
    hover_data=pandas_data.columns,
    title='K-Means聚类结果散点图 (PCA降维)',
    color_continuous_scale=px.colors.qualitative.Set1,
    width=800,
    height=600
)

# 添加聚类中心
fig_cluster.add_scatter(
    x=centers_pd['pca_x'],
    y=centers_pd['pca_y'],
    mode='markers',
    marker=dict(size=15, color='black', symbol='x'),
    name='聚类中心'
)

cluster_scatter_img = fig_cluster.to_html(full_html=False)

# 5. 聚类分布条形图
cluster_counts = pandas_data['prediction'].value_counts().sort_index()
# 修改聚类分布条形图的颜色
fig_bar = px.bar(
    cluster_counts,
    x=cluster_counts.index,
    y=cluster_counts.values,
    title='聚类分布情况',
    text=cluster_counts.values,
    color=cluster_counts.index,
    color_discrete_sequence=px.colors.qualitative.Set1  # 使用定性颜色序列
)
fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
cluster_bar_img = fig_bar.to_html(full_html=False)

# 6. 新增：事故时间分布热力图
df['hour'] = df['hour'].astype(int)
hour_severity = df.groupby(['hour', 'Accident_Severity']).size().unstack(fill_value=0)
fig_heatmap = px.imshow(
    hour_severity.T,
    labels=dict(x="小时", y="严重程度", color="事故数量"),
    x=hour_severity.index,
    y=hour_severity.columns,
    title='不同时段事故严重程度热力图',
    color_continuous_scale='Blues'
)
heatmap_img = fig_heatmap.to_html(full_html=False)

# 7. 新增：事故严重程度与特征关系平行坐标图
# 选择部分特征和样本用于可视化
# 将分类标签映射为数值
sample_df = df.sample(frac=0.3).sort_values('Accident_Severity')
features_for_parallel = ['hour', 'is_peak_hour', 'month'] + [col for col in df.columns if col.endswith('_indexed')][:3]
sample_df['color_value'] = sample_df['Accident_Severity'].map({
    'Fatal': 0,
    'Serious': 1,
    'Slight': 2
    # 添加其他可能的严重程度级别
})

# 创建平行坐标图
fig_parallel = px.parallel_coordinates(
    sample_df,
    dimensions=features_for_parallel + ['Accident_Severity'],
    color='color_value',  # 使用数值列作为颜色
    color_continuous_scale=px.colors.sequential.Viridis,
    title='事故严重程度与特征关系平行坐标图'
)
parallel_img = fig_parallel.to_html(full_html=False)

# 8. 新增：交互式3D散点图（选择部分特征）
if len(feature_cols) >= 3:
    # 确保选择的是数据框中存在的列
    selected_features = [col for col in feature_cols if col in pandas_data.columns][:3]

    # 如果找到足够的特征，创建3D散点图
    if len(selected_features) >= 3:
        fig_3d = px.scatter_3d(
            pandas_data.sample(1000),
            x=selected_features[0],
            y=selected_features[1],
            z=selected_features[2],
            color='prediction',
            title='3D特征空间中的聚类分布',
            hover_name='prediction'
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
        scatter_3d_img = fig_3d.to_html(full_html=False)
    else:
        scatter_3d_img = None
else:
    scatter_3d_img = None

# ----------------------
# 创建HTML报告
# ----------------------
html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>交通事故数据分析报告</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            padding: 20px 0;
            background-color: #1a5276;
            color: white;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .section {
            background-color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .section-title {
            color: #1a5276;
            border-bottom: 2px solid #1a5276;
            padding-bottom: 10px;
            margin-top: 0;
        }
        .metrics-container {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #eaf2f8;
            padding: 15px;
            border-radius: 5px;
            width: 45%;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .metric-title {
            font-weight: bold;
            color: #2874a6;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            color: #1a5276;
        }
        .image-container {
            text-align: center;
            margin: 20px 0;
        }
        .image-container div {
            display: inline-block;
            margin: 0 auto;
        }
        .image-caption {
            font-style: italic;
            color: #666;
            margin-top: 5px;
        }
        .feature-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .feature-table th, .feature-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .feature-table th {
            background-color: #eaf2f8;
        }
        .feature-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 14px;
        }
        .interactive-warning {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 4px;
            margin: 15px 0;
            border-left: 4px solid #ffc107;
        }
        .filter-container {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            border: 1px solid #eee;
        }
        .filter-title {
            font-weight: bold;
            margin-bottom: 10px;
            color: #1a5276;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>交通事故数据分析报告</h1>
        <p>基于机器学习模型和聚类分析的事故特征研究</p>
    </div>

    <div class="section">
        <h2 class="section-title">数据概览</h2>
        <p>本分析基于交通事故数据集，包含 {{ data_shape[0] }} 条记录和 {{ data_shape[1] }} 个特征。</p>

        <div class="image-container">
            {{ severity_pie_img|safe }}
            <p class="image-caption">图1: 事故严重程度分布</p>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">多分类模型分析</h2>
        <p>使用随机森林模型预测事故严重程度，评估结果如下：</p>

        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-title">准确率</div>
                <div class="metric-value">{{ "%.2f"|format(accuracy*100) }}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">F1分数</div>
                <div class="metric-value">{{ "%.2f"|format(f1_score*100) }}%</div>
            </div>
        </div>

        <div class="image-container">
            {{ multi_feature_img|safe }}
            <p class="image-caption">图2: 多分类模型特征重要性排名</p>
        </div>

        <h3>Top 10重要特征</h3>
        <table class="feature-table">
            <tr>
                <th>排名</th>
                <th>特征</th>
                <th>重要性</th>
            </tr>
            {% for idx, row in feature_importance.head(10).iterrows() %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ row['特征'] }}</td>
                <td>{{ "%.4f"|format(row['重要性']) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">二分类模型分析</h2>
        <p>使用随机森林模型预测是否为严重事故，评估结果如下：</p>

        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-title">AUC</div>
                <div class="metric-value">{{ "%.4f"|format(auc) }}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">PR曲线下面积</div>
                <div class="metric-value">{{ "%.4f"|format(pr) }}</div>
            </div>
        </div>

        <div class="image-container">
            {{ binary_feature_img|safe }}
            <p class="image-caption">图3: 二分类模型特征重要性排名</p>
        </div>

        <h3>Top 10重要特征</h3>
        <table class="feature-table">
            <tr>
                <th>排名</th>
                <th>特征</th>
                <th>重要性</th>
            </tr>
            {% for idx, row in bin_feature_importance.head(10).iterrows() %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ row['特征'] }}</td>
                <td>{{ "%.4f"|format(row['重要性']) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">聚类分析</h2>
        <p>使用K-Means算法对事故数据进行聚类分析，共发现3个主要聚类：</p>

        <div class="image-container">
            {{ cluster_bar_img|safe }}
            <p class="image-caption">图4: 聚类分布情况</p>
        </div>

        <div class="image-container">
            {{ cluster_scatter_img|safe }}
            <p class="image-caption">图5: K-Means聚类结果散点图 (PCA降维)</p>
        </div>

        <h3>聚类中心特征</h3>
        <table class="feature-table">
            <tr>
                <th>聚类ID</th>
                <th>主要特征</th>
            </tr>
            {% for i in range(3) %}
            <tr>
                <td>聚类 {{ i+1 }}</td>
                <td>{{ cluster_centers[i] }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">高级分析</h2>

        <div class="image-container">
            {{ heatmap_img|safe }}
            <p class="image-caption">图6: 不同时段事故严重程度热力图</p>
        </div>

        <div class="image-container">
            {{ parallel_img|safe }}
            <p class="image-caption">图7: 事故严重程度与特征关系平行坐标图</p>
        </div>

        {% if scatter_3d_img %}
        <div class="image-container">
            {{ scatter_3d_img|safe }}
            <p class="image-caption">图8: 3D特征空间中的聚类分布</p>
        </div>
        {% endif %}
    </div>

    <div class="footer">
        <p>报告生成时间: {{ current_time }}</p>
        <p>© 2023 交通事故分析团队 - 所有权利保留</p>
    </div>
</body>
</html>
"""

# 准备聚类中心描述
cluster_centers = []
for i, center in enumerate(centers_scaled):
    # 获取最重要的5个特征
    top_indices = np.argsort(center)[-5:][::-1]
    top_features = [feature_cols[idx] for idx in top_indices]
    cluster_centers.append(", ".join(top_features))

# 准备数据形状信息
data_shape = df.shape

# 获取当前时间
from datetime import datetime
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 渲染HTML
html_content = Template(html_template).render(
    severity_pie_img=severity_pie_img,
    multi_feature_img=multi_feature_img,
    binary_feature_img=binary_feature_img,
    cluster_bar_img=cluster_bar_img,
    cluster_scatter_img=cluster_scatter_img,
    heatmap_img=heatmap_img,
    parallel_img=parallel_img,
    scatter_3d_img=scatter_3d_img,
    accuracy=accuracy,
    f1_score=f1_score,
    auc=auc,
    pr=pr,
    feature_importance=feature_importance,
    bin_feature_importance=bin_feature_importance,
    cluster_centers=cluster_centers,
    data_shape=data_shape,
    current_time=current_time
)

# 保存HTML文件
with open("traffic983.html", "w", encoding="utf-8") as f:
    f.write(html_content)
print("\n交互式HTML报告已生成: traffic983.html")

# 停止Spark
spark.stop()