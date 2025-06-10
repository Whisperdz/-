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

# ----------------------
# 环境配置
# ----------------------
os.environ['PYSPARK_PYTHON'] = '/root/.virtualenvs/pythonProject4/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/root/.virtualenvs/pythonProject4/bin/python3'
java_home = "/export/server/jdk1.8.0_241/"
os.environ['JAVA_HOME'] = java_home
os.environ['PATH'] = f"{java_home}/bin:{os.environ['PATH']}"

# ----------------------
# 数据读取与清洗
# ----------------------
try:
    df = pd.read_csv('traffic.csv')
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

spark_df = spark.createDataFrame(df)

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

# 输出聚类中心（需将标准化后的中心通过PCA转换到二维空间）
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
# 可视化部分（确保所有图表使用指定字体）
# ----------------------
# 设置图形风格（已在字体设置中为seaborn设置过字体）

# 1. 多分类模型特征重要性图
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
ax = sns.barplot(x="重要性", y="特征", data=top_features, palette="viridis")
plt.title("多分类模型特征重要性排名", fontproperties=font_prop)
plt.xlabel("重要性", fontproperties=font_prop)
plt.ylabel("特征", fontproperties=font_prop)
plt.tight_layout()
plt.savefig("multiclass_feature_importance.png")
plt.show()

# 2. 二分类模型特征重要性图
plt.figure(figsize=(12, 8))
top_bin_features = bin_feature_importance.head(10)
ax = sns.barplot(x="重要性", y="特征", data=top_bin_features, palette="magma")
plt.title("二分类模型特征重要性排名", fontproperties=font_prop)
plt.xlabel("重要性", fontproperties=font_prop)
plt.ylabel("特征", fontproperties=font_prop)
plt.tight_layout()
plt.savefig("binary_feature_importance.png")
plt.show()

# 3. 聚类分析散点图
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="pca_x",
    y="pca_y",
    hue="prediction",
    palette="Set1",
    s=100,
    alpha=0.7,
    data=pandas_data
)
plt.title("K-Means聚类结果散点图 (PCA降维)", fontproperties=font_prop)
plt.xlabel(f"PCA1 ({pca_model.explainedVariance[0]:.2%})", fontproperties=font_prop)
plt.ylabel(f"PCA2 ({pca_model.explainedVariance[1]:.2%})", fontproperties=font_prop)

# 添加聚类中心
plt.scatter(
    centers_pd["pca_x"],
    centers_pd["pca_y"],
    s=200,
    marker='X',
    c='black',
    label='聚类中心'
)

plt.legend(title='聚类类别', prop=font_prop)  # 为图例设置字体
plt.tight_layout()
plt.savefig("clustering_scatter.png")
plt.show()

# 4. 事故严重程度分布饼图
plt.figure(figsize=(8, 8))
severity_counts = df['Accident_Severity'].value_counts()
plt.pie(severity_counts, labels=severity_counts.index, autopct='%1.1f%%',
        startangle=90, colors=sns.color_palette('pastel'), textprops={'fontproperties': font_prop})
plt.title('事故严重程度分布', fontproperties=font_prop)
plt.tight_layout()
plt.savefig("severity_distribution.png")
plt.show()

# 5. 聚类分布条形图
plt.figure(figsize=(10, 6))
cluster_counts = pandas_data['prediction'].value_counts().sort_index()
ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="Blues_d")
plt.title('聚类分布情况', fontproperties=font_prop)
plt.xlabel('聚类类别', fontproperties=font_prop)
plt.ylabel('样本数量', fontproperties=font_prop)
plt.xticks(rotation=0)

# 添加数值标签
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                fontproperties=font_prop,
                fontsize=10, color='black',
                xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig("cluster_distribution.png")
plt.show()

# 停止Spark
spark.stop()