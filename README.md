# 数据分析与数据挖掘课程项目：电信客户流失预测与价值细分系统

## 1. 项目简介
本项目基于 IBM/Kaggle Telco Customer Churn 数据集，构建“流失预警 + 客户价值细分”双模块系统。项目覆盖数据清洗、探索性分析、特征工程、类别不平衡处理、分类建模、模型评估、特征重要性解释与客户分群。

## 2. 项目结构
```text
项目六_电信客户流失预测与价值细分系统/
├── data/
│   ├── raw/                         # 原始数据
│   └── processed/                   # 清洗后的数据
├── src/
│   └── run_analysis.py              # 一键复现实验脚本
├── outputs/
│   ├── figures/                     # EDA、ROC/PR、混淆矩阵、聚类图
│   └── tables/                      # 指标表、特征重要性、聚类统计
├── docs/                            # 报告与补充材料
├── ppt/                             # 答辩 PPT
├── Telco_Churn_Mining.ipynb         # 完整 Notebook
├── requirements.txt                 # 依赖环境
└── README.md                        # 运行说明
```

## 3. 数据来源
数据集：Telco Customer Churn，共 7043 条样本、21 个原始字段。目标变量为 `Churn`，表示客户是否流失。

本提交包已包含课程实验所需 CSV：
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

## 4. 环境安装
建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

## 5. 一键运行
在项目根目录执行：

```bash
python src/run_analysis.py
```

或打开 Notebook：

```bash
jupyter notebook Telco_Churn_Mining.ipynb
```

## 6. 输出结果
运行后自动生成：
- `outputs/tables/model_metrics.csv`：分类模型测试集指标
- `outputs/tables/cv_metrics.csv`：交叉验证 ROC-AUC
- `outputs/tables/feature_importance_top20.csv`：Top 特征重要性
- `outputs/tables/cluster_summary.csv`：客户分群统计
- `outputs/figures/roc_curves.png`：ROC 曲线
- `outputs/figures/pr_curves.png`：PR 曲线
- `outputs/figures/confusion_matrix_best.png`：混淆矩阵
- `outputs/figures/customer_segments.png`：客户分群可视化

## 7. 主要实验结果
在固定随机种子 42、分层 8:2 训练测试划分下，Gradient Boosting 模型取得最佳 ROC-AUC：约 0.847。为业务召回优先目标，使用 F1 最优阈值 0.280 后，召回率提升至约 0.794，F1 提升至约 0.637。

## 8. 可复现性说明
- 固定随机种子：42
- 无硬编码绝对路径，脚本以项目根目录为基准
- 所有图表和指标均由脚本自动生成
- 代码遵循 PEP8，关键函数均包含 docstring
