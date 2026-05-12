# 电信客户流失预测与价值细分

基于机器学习的电信客户流失预测与价值细分系统，覆盖从原始数据到可操作洞察的完整流程：数据清洗、探索性分析、分类建模、阈值优化、特征重要性解释与 K-Means 客户分群。

## 项目简介

客户流失是电信行业的核心业务问题。本项目构建了一个双模块系统：

- **流失预测** — 预测客户是否会流失，通过阈值调优最大化业务召回率
- **客户分群** — 将客户划分为四个行为群体，为定向留存策略提供依据

**数据集**：[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 位客户，21 个特征，流失率 26.5%。

## 实验结果

| 模型 | ROC-AUC | F1（阈值=0.28） | 召回率 |
|---|---|---|---|
| **Gradient Boosting** | **0.847** | **0.637** | **0.794** |
| Random Forest | 0.846 | 0.634 | 0.765 |
| Logistic Regression + SMOTE | 0.844 | 0.622 | 0.789 |

交叉验证 ROC-AUC（3 折分层）：**0.848 ± 0.008**

**Top 预测特征**：`ChargePerTenure`、`Contract_Month-to-month`、`InternetService_Fiber optic`

**客户分群结果**（K-Means，k=4）：

| 分群 | 客户数 | 流失率 | 平均使用月数 | 平均月费 |
|---|---|---|---|---|
| 高风险新客户 | 755 | 66.9% | 1.7 个月 | $69.73 |
| 中风险成长客户 | 1,987 | 39.3% | 19.4 个月 | $78.71 |
| 稳定高价值客户 | 2,061 | 14.5% | 59.9 个月 | $89.96 |
| 低消费忠实客户 | 2,240 | 12.8% | 28.9 个月 | $27.53 |

## 项目结构

```
telco_project/
├── data/
│   └── raw/                        # 原始数据集（已包含）
├── src/
│   └── run_analysis.py             # 可复现的端到端分析脚本
├── outputs/
│   ├── figures/                    # EDA、ROC/PR 曲线、混淆矩阵、分群图
│   └── tables/                     # 模型指标、特征重要性、分群统计
├── Telco_Churn_Mining.ipynb        # 含完整说明的 Notebook
└── requirements.txt
```

## 快速开始

**环境要求**：Python 3.10+

```bash
pip install -r requirements.txt
python src/run_analysis.py
```

所有图表和表格自动输出到 `outputs/`。Notebook `Telco_Churn_Mining.ipynb` 提供相同的流程并附有逐步说明。

## 技术细节

### 特征工程

在原始 21 个字段基础上新增三个衍生特征：

- `ChargePerTenure` — 月费除以 `(tenure + 1)`，衡量新客户的费用压力
- `AvgMonthlyCharge` — `TotalCharges / tenure`，对零租期客户回退到 `MonthlyCharges`
- `TenureSegment` — 按客户生命周期分段（0–6、7–12、13–24、25–48、49–72 个月）

### 类别不平衡处理

数据集存在类别不平衡（流失率 26.5%），采用两种策略对比：

- SMOTE 过采样（用于逻辑回归管道）
- `class_weight="balanced"`（用于随机森林、梯度提升）

### 阈值优化

默认阈值（0.5）偏向准确率但召回率不足。通过遍历精确率-召回率曲线选取 F1 最优阈值，最终确定 **0.28**，将召回率从 51% 提升至 **79%**。

### 客户分群

对 `[tenure, MonthlyCharges, TotalCharges, ChargePerTenure]` 标准化后进行 K-Means（k=4）聚类，按流失率排序，输出可直接用于运营决策的优先级留存列表。

## 可复现性

- 全局随机种子固定为 `42`
- 无硬编码绝对路径，脚本以项目根目录为基准自动解析路径
- 每次运行从零重新生成全部输出
