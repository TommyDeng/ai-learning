import joblib
import pandas as pd

model = joblib.load("pass_model.joblib")

new_data = pd.DataFrame([[4, 15], [7, 40]], columns=["study_hours", "practice_count"])

preds = model.predict(new_data)

for i, p in enumerate(preds):
    result = "及格" if p == 1 else "不及格"
    print(f"学生{i + 1}: {result}")


# 1. 加载模型文件
model = joblib.load("pass_model.joblib")

# 2. 查看模型的类型和参数配置
print("模型配置:", model)

# 3. 查看训练学到的权重系数 (Coef / Weights)
# 对应特征：[study_hours, practice_count] 前面乘的系数
print("特征权重系数 (w):", model.coef_)

# 4. 查看偏置项/截距 (Intercept / Bias)
print("截距项 (b):", model.intercept_)

# 5. 查看模型训练时用到的特征数量
print("特征数量:", model.n_features_in_)
