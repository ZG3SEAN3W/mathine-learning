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

# --------------------------------------------
# 1. 数据加载和预处理
# --------------------------------------------

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


# --------------------------------------------
# 2. 数据清洗
# --------------------------------------------

def extract_z_zz(value):
    if pd.isna(value): return (np.nan, "")
    match = re.match(r"([+-]?\d+\.\d+)\s*(\(\w+\))?", str(value))
    return (float(match.group(1)), match.group(2).strip("()") if match.group(2) else "") if match else (np.nan, "")


df[["Z_zz_value", "Anion_type"]] = df["Anion_Z_zz"].apply(
    lambda x: pd.Series(extract_z_zz(x)))
df["Anion_type"] = df["Anion_type"].str.replace("₁", "")

anion_avg = {
    "Cl": -1.47, "Br": -1.26, "I": -1.07,
    "Se": -2.82, "Te": -3.23, "N": -4.99, "S": -1.85
}
df["Z_zz_value"] = df.apply(
    lambda row: anion_avg.get(row["Anion_type"], np.nan)
    if pd.isna(row["Z_zz_value"]) else row["Z_zz_value"], axis=1)

df = df.drop(["Anion_Z_zz", "Anion_type"], axis=1)
df = df[~df["Material"].isin(["rs-BeO", "BiFeO₃", "SnSe"])]

# --------------------------------------------
# 3. 特征工程
# --------------------------------------------

numerical_features = ["a", "c", "Interlayer_spacing", "Band_gap", "CBM-VBM_difference",
                      "ω_TO", "Z_TO", "Bi_Z_xx", "Z_zz_value"]
categorical_features = ["Space_group"]
targets = ["ε_c", "P"]

# --------------------------------------------
# 4. 预处理管道
# --------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --------------------------------------------
# 5. 模型训练
# --------------------------------------------

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42)
}

trained_models = {}
for model_name, model in models.items():
    print(f"\nTraining {model_name} model...")
    X = df.drop(targets + ["Material"], axis=1)
    y = df[targets]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('regressor', MultiOutputRegressor(model))
    ])

    pipeline.fit(X_train, y_train)
    trained_models[model_name] = pipeline

    # 评估模型
    y_pred = pipeline.predict(X_test)
    print(f"{model_name} ε_c R²: {r2_score(y_test['ε_c'], y_pred[:, 0]):.3f}")
    print(f"{model_name} P R²: {r2_score(y_test['P'], y_pred[:, 1]):.3f}")

# SVR模型训练
svr_pipelines = {}
for target in targets:
    print(f"\nTraining SVR for {target}...")
    X = df.drop(targets + ["Material"], axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline(steps=[
        ('prep', preprocessor),
        ('regressor', SVR(kernel='rbf'))
    ])

    pipeline.fit(X_train, y_train)
    svr_pipelines[target] = pipeline

    # 评估模型
    y_pred = pipeline.predict(X_test)
    print(f"SVR {target} R²: {r2_score(y_test, y_pred):.3f}")


# --------------------------------------------
# 6. SHAP分析
# --------------------------------------------

def generate_shap_summary(model_name, pipeline, X_train):
    try:
        # 获取特征名称
        preprocessor = pipeline.named_steps['prep']
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = numerical_features + list(cat_features)

        # 预处理数据
        X_transformed = preprocessor.transform(X_train)

        # 获取模型
        model = pipeline.named_steps['regressor']

        # 多输出模型处理
        if isinstance(model, MultiOutputRegressor):
            for i, estimator in enumerate(model.estimators_):
                print(f"  Processing target {targets[i]}...")

                # 树模型使用TreeExplainer
                if hasattr(estimator, 'tree_') or hasattr(estimator, 'get_booster'):
                    explainer = shap.TreeExplainer(estimator)
                    shap_values = explainer.shap_values(X_transformed)
                else:
                    explainer = shap.KernelExplainer(estimator.predict, shap.sample(X_transformed, 50))
                    shap_values = explainer.shap_values(X_transformed)

                # 绘制图形
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_transformed,
                                  feature_names=feature_names,
                                  show=False)
                plt.title(f"{model_name} - {targets[i]}")
                plt.tight_layout()
                plt.savefig(f"{model_name}_{targets[i]}_shap.png", dpi=300)
                plt.close()

        # 单输出模型处理
        else:
            print(f"  Processing single target...")
            if hasattr(model, 'tree_') or hasattr(model, 'get_booster'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X_transformed, 50))
                shap_values = explainer.shap_values(X_transformed)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_transformed,
                              feature_names=feature_names,
                              show=False)
            plt.title(model_name)
            plt.tight_layout()
            plt.savefig(f"{model_name}_shap.png", dpi=300)
            plt.close()

    except Exception as e:
        print(f"Error in {model_name}: {str(e)}")


# 执行SHAP分析
print("\nStarting SHAP analysis...")
for model_name, pipeline in trained_models.items():
    print(f"\nAnalyzing {model_name}:")
    X_full = df.drop(targets + ["Material"], axis=1)
    X_train_full, _, _, _ = train_test_split(X_full, df[targets], test_size=0.2, random_state=42)
    generate_shap_summary(model_name, pipeline, X_train_full)

for target, pipeline in svr_pipelines.items():
    print(f"\nAnalyzing SVR_{target}:")
    X_svr = df.drop(targets + ["Material"], axis=1)
    X_train_svr, _, _, _ = train_test_split(X_svr, df[target], test_size=0.2, random_state=42)
    generate_shap_summary(f"SVR_{target}", pipeline, X_train_svr)

print("\nAll analyses completed! Check generated PNG files.")