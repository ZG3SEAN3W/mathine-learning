import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
import shap
import matplotlib.pyplot as plt
import re
from sklearn.multioutput import MultiOutputRegressor
import os

# ============================================
# 1. 数据加载和预处理
# ============================================

data = [
    ["BiOCl", 3.89, 12.3, 6.15, "P6₃/mmc", 3.67, 3.6, 10.2, 0.85, 5.34, "-1.47 (Cl)", 3, 25.2],
    ["BiOBr", 3.92, 12.45, 6.23, "P6₃/mmc", 3.12, 3, 9.8, 0.82, 5.59, "-1.23 (Br)", 2.8, 23.4],
    ["BiOI", 4.01, 12.8, 6.4, "P4/mmm", 2.37, 2.4, 8.5, 0.78, 5.92, "-1.06 (I)", 1.8, 20.7],
    ["Bi₂O₂Se", 3.89, 13.1, 6.55, "R3m", 1.17, 1.2, 6.7, 0.92, 6.1, "-3.03 (Se)", 1.7, 33.6],
    ["Bi₂O₂CN₂", 3.76, 11.9, 5.95, "I4/mmm", 1.93, 1.8, 7.5, 0.88, 5.14, "-4.99 (N)", 1.2, 28.9],
    ["BiCuOSe", 3.92, 12.6, 6.3, "P4/nmm", 1.25, 1, 6.9, 0.95, 6.36, "-2.61 (Se)", 3.4, 30.3],
    ["Bi₂OS₂", 3.97, 12.7, 6.35, "P2₁/c", 1.33, 1.1, 10.8, 0.81, 6.57, "-1.85 (S₁)", 1.4, 27.4],
    ["Bi₅O₄S₃Cl", 3.92, 13.5, 6.75, "C2/m", 0.84, 0.8, 2.1, 1.12, 7.37, "-2.82 (Cl)", 0.6, 23.1],
    ["Bi₂O₂Te", 4.05, 13.2, 6.6, "R3m", 0.95, 0.9, 5.3, 1.05, 6.8, "-3.20 (Te)", 0.9, 35],
    ["PbFCl", 4.09, 12.4, 6.2, "P4/nmm", 4.59, 4.3, 8.5, 0.75, 3.71, "-1.38 (Cl)", 2.6, 16.4],
    ["Bi₂O₂Br", 3.95, 12.5, 6.25, "P6₃/mmc", 2.85, 2.7, 9.2, 0.82, 5.48, "-1.30 (Br)", 2.5, 24.5],
    ["Bi₂O₂Te(2)", 4.08, 13.3, 6.65, "R3m", 0.92, 0.9, 5.5, 0.95, 6.75, "-3.25 (Te)", 0.9, 35],
    ["Bi₂O₂I", 4.1, 12.9, 6.45, "P4/mmm", 2.1, 2.0, 8, 0.78, 5.9, "-1.15 (I)", 1.5, 21],
    ["PbFBr", 4.22, 12.6, 6.3, "P4/nmm", 3.3, 3.1, 7.8, 0.75, 3.85, "-1.25 (Br)", 2.8, 16.8],
    ["PbFI", 4.25, 12.8, 6.4, "P4/nmm", 2.95, 2.8, 7.5, 0.72, 3.9, "-1.10 (I)", 2.3, 18],
    ["rs-BeO", 4.21, np.nan, np.nan, "Fm-3m", 10.6, 9.8, 8.3, 0.95, np.nan, np.nan, 0.5, np.nan],
    ["BiFeO₃", 3.96, 4.01, np.nan, "R3c", 2.7, 2.5, 15.2, 1.2, np.nan, np.nan, 2.0, 100],
    ["SnSe", 11.49, 4.44, 2.22, "Pnma", 0.9, 0.8, 7.5, 0.75, np.nan, np.nan, 2.8, np.nan]
]

columns = ["Material", "a", "c", "Interlayer_spacing", "Space_group", "Band_gap", "CBM-VBM_difference",
           "ω_TO", "Z_TO", "Bi_Z_xx", "Anion_Z_zz", "ε_c", "P"]
df = pd.DataFrame(data, columns=columns)

# 在模型训练部分之前定义输出路径
output_dir = r"F:\sicau_add_learning\西湖大学-李文彬"
os.makedirs(output_dir, exist_ok=True)

# ============================================
# 2. 数据清洗
# ============================================

def extract_z_zz(value):
    """提取Anion_Z_zz中的数值和阴离子类型"""
    if pd.isna(value): return (np.nan, "")
    match = re.match(r"([+-]?\d+\.\d+)\s*(\(\w+\))?", str(value))
    return (float(match.group(1)), match.group(2).strip("()") if match.group(2) else "") if match else (np.nan, "")


# 拆分Anion_Z_zz列
df[["Z_zz_value", "Anion_type"]] = df["Anion_Z_zz"].apply(
    lambda x: pd.Series(extract_z_zz(x)))
df["Anion_type"] = df["Anion_type"].str.replace("₁", "")

# 填充缺失值
anion_avg = {
    "Cl": -1.47, "Br": -1.26, "I": -1.07,
    "Se": -2.82, "Te": -3.23, "N": -4.99, "S": -1.85
}
df["Z_zz_value"] = df.apply(
    lambda row: anion_avg.get(row["Anion_type"], np.nan)
    if pd.isna(row["Z_zz_value"]) else row["Z_zz_value"], axis=1)

# 清理不需要的列
df = df.drop(["Anion_Z_zz", "Anion_type"], axis=1)
df = df[~df["Material"].isin(["rs-BeO", "BiFeO₃", "SnSe"])]

# ============================================
# 3. 特征工程
# ============================================

numerical_features = ["a", "c", "Interlayer_spacing", "Band_gap", "CBM-VBM_difference",
                      "ω_TO", "Z_TO", "Bi_Z_xx", "Z_zz_value"]
categorical_features = ["Space_group"]
targets = ["ε_c", "P"]

# ============================================
# 4. 预处理管道
# ============================================

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# ============================================
# 5. 模型训练与预测（修改版：全样本预测）
# ============================================

# 初始化存储
trained_models = {}
svr_pipelines = {}
all_predictions = pd.DataFrame()

# 多输出模型
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}


# 全样本预测函数
def predict_all_samples(pipeline, X, model_name):
    """使用训练好的模型预测所有样本"""
    y_pred = pipeline.predict(X)
    return pd.DataFrame({
        'Material': df['Material'],
        f'{model_name}_ε_c_true': df['ε_c'],
        f'{model_name}_ε_c_pred': y_pred[:, 0],
        f'{model_name}_P_true': df['P'],
        f'{model_name}_P_pred': y_pred[:, 1]
    })


for model_name, model in models.items():
    print(f"\n=== Training & Predicting {model_name} ===")
    X_all = df.drop(targets + ["Material"], axis=1)
    y_all = df[targets]

    # 构建管道
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('regressor', MultiOutputRegressor(model))
    ])

    # 全样本训练
    pipeline.fit(X_all, y_all)
    trained_models[model_name] = pipeline

    # 全样本预测
    model_pred = predict_all_samples(pipeline, X_all, model_name)

    # 合并预测结果
    if all_predictions.empty:
        all_predictions = model_pred
    else:
        all_predictions = all_predictions.merge(model_pred, on='Material')

    # 输出评估指标
    print(f"{model_name} ε_c R²: {r2_score(y_all['ε_c'], model_pred[f'{model_name}_ε_c_pred']):.3f}")
    print(f"{model_name} P R²: {r2_score(y_all['P'], model_pred[f'{model_name}_P_pred']):.3f}")


# SVR模型（单目标）
def predict_svr_all(pipeline, X, target):
    """SVR全样本预测"""
    y_pred = pipeline.predict(X)
    return pd.DataFrame({
        'Material': df['Material'],
        f'SVR_{target}_true': df[target],
        f'SVR_{target}_pred': y_pred
    })


for target in targets:
    print(f"\n=== Training & Predicting SVR for {target} ===")
    X_all = df.drop(targets + ["Material"], axis=1)
    y_all = df[target]

    # 构建管道
    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('regressor', SVR(kernel='rbf'))
    ])

    # 全样本训练
    pipeline.fit(X_all, y_all)
    svr_pipelines[target] = pipeline

    # 全样本预测
    svr_pred = predict_svr_all(pipeline, X_all, target)

    # 合并预测结果
    all_predictions = all_predictions.merge(svr_pred, on='Material')

    # 输出评估指标
    print(f"SVR {target} R²: {r2_score(y_all, svr_pred[f'SVR_{target}_pred']):.3f}")

# 保存完整预测结果
output_path = os.path.join(output_dir, 'all_samples_predictions.csv')
all_predictions.to_csv(output_path, index=False, float_format='%.4f')
print(f"\n所有样本预测结果已保存至 {output_path}")


# ============================================
# 6. SHAP分析（全样本分析）
# ============================================

# 修改后的SHAP分析函数
def generate_full_shap(model_name, pipeline):
    """全样本SHAP分析"""
    try:
        # 获取预处理信息
        preprocessor = pipeline.named_steps['prep']
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = numerical_features + list(cat_features)

        # 全样本数据
        X_all = df.drop(targets + ["Material"], axis=1)
        X_transformed = preprocessor.transform(X_all)

        # 获取模型
        model = pipeline.named_steps['regressor']

        # 多输出处理
        if isinstance(model, MultiOutputRegressor):
            for i, estimator in enumerate(model.estimators_):
                print(f"  Generating SHAP for {targets[i]}...")

                # 树模型解释器
                if hasattr(estimator, 'tree_') or hasattr(estimator, 'get_booster'):
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_transformed)
                else:
                    explainer = shap.KernelExplainer(estimator.predict, shap.sample(X_transformed, 50))
                    shap_values = explainer.shap_values(X_transformed)

                # 绘图保存
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_transformed,
                                  feature_names=feature_names,
                                  show=False)
                plt.title(f"{model_name} - {targets[i]} (All Samples)")
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"{model_name}_{targets[i]}_full_shap.png"), dpi=300)
                plt.close()

        # 单输出处理
        else:
            print("  Generating SHAP for single target...")
            if hasattr(model, 'tree_') or hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_transformed, 50))
                shap_values = explainer.shap_values(X_transformed)

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_transformed,
                              feature_names=feature_names,
                              show=False)
            plt.title(f"{model_name} (All Samples)")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{model_name}_full_shap.png"), dpi=300)
            plt.close()

    except Exception as e:
        print(f"SHAP分析失败: {str(e)}")


# 执行全样本SHAP分析
print("\n=== 开始全样本SHAP分析 ===")
for model_name, pipeline in trained_models.items():
    print(f"\n分析模型: {model_name}")
    generate_full_shap(model_name, pipeline)

for target, pipeline in svr_pipelines.items():
    print(f"\n分析模型: SVR_{target}")
    generate_full_shap(f"SVR_{target}", pipeline)

print("\n=== 全样本分析完成 ===")