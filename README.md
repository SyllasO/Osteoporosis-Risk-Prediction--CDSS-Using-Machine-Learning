# Osteoporosis-Risk-Prediction--CDSS-Using-Machine-Learning
Implementation of a Machine Learning–Based Clinical Decision Support System for Osteoporosis Risk Prediction

# 🦴 Osteoporosis Risk Prediction using Machine Learning  
*A Clinical Decision Support System (CDSS) for Early Detection*

---

## 📌 Project Overview

Osteoporosis affects over **200 million people worldwide**, contributing to **8.9 million fractures annually**. Early detection is essential, yet DXA scans remain underused due to cost, accessibility barriers, and inconsistent screening practices.

This project develops a **Machine Learning–based Clinical Decision Support System (CDSS)** that predicts osteoporosis risk using demographic, lifestyle, and nutritional features. The workflow includes preprocessing, model comparison, explainable AI (SHAP), and clinical utility evaluation (Decision Curve Analysis, DCA).[web:5][web:46]

The system supports **early identification** of at-risk individuals and demonstrates how machine learning can enhance preventive healthcare.[web:49]

---

## 📂 Dataset

**Source:** Kaggle – *Lifestyle Factors Influencing Osteoporosis*  
🔗 https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis/data[web:46][web:47]

- **Records:** 1,958  
- **Features:** 15  
- **Target:** `Osteoporosis` (0 = No, 1 = Yes)

### Key Features

| Feature           | Type         | Description                          |
|------------------|-------------|--------------------------------------|
| Age              | Numeric     | Primary determinant of bone loss     |
| Gender           | Categorical | Women at higher risk                 |
| Family History   | Binary      | Genetic risk indicator               |
| Hormonal Changes | Binary      | Menopause-related effects            |
| Body Weight      | Numeric     | Low weight increases risk            |
| Calcium Intake   | Numeric     | Nutrient affecting bone strength     |
| Vitamin D Intake | Numeric     | Supports calcium absorption          |
| Physical Activity| Categorical | Protects bone density                |
| Prior Fractures  | Binary      | Strong clinical predictor            |

Zero-variance features such as `Alcohol Consumption` and `Medications` were removed to avoid redundant information.[web:49]

---

## 🧹 Data Preprocessing

- Removed duplicates and validated dataset integrity  
- Median imputation for numeric features  
- Mode imputation for categorical variables  
- One-hot encoding for categorical fields  
- Standardization (e.g., `StandardScaler`) applied to numeric features  
- Target encoded as binary (`0` = No osteoporosis, `1` = Osteoporosis)  
- SHAP used to evaluate feature relevance and biomedical consistency[web:5][web:49]

---

## 🤖 Models Trained

Seven machine learning models were developed:

1. Logistic Regression  
2. Random Forest  
3. Gradient Boosting Classifier ⭐ **Best Model**  
4. Support Vector Machine (RBF kernel)  
5. K-Nearest Neighbors  
6. Naïve Bayes  
7. Deep Neural Network (optional / experimental)

---

## 🧪 Model Evaluation

Models were evaluated using:

- Accuracy  
- Precision & Recall  
- F1-score  
- Confusion Matrix  
- ROC Curve & AUROC  
- Calibration Curve  
- Decision Curve Analysis (DCA)  
- SHAP-based explainability[web:5]

### ⭐ Best Model: Gradient Boosting Classifier

The Gradient Boosting Classifier was selected as the final model due to:

- High AUROC  
- Well-calibrated risk probabilities  
- Superior net clinical benefit on Decision Curve Analysis  
- Clinically meaningful SHAP explanations that match known osteoporosis risk factors[web:5][web:51]

---

## 📊 Visualizations Included

The project generates and saves the following plots:

- Confusion Matrix  
- ROC Curves  
- Calibration Plot  
- Decision Curve Analysis Plot  
- SHAP Summary Plot  
- SHAP Force Plot (individual prediction)  
- Age and Calcium Intake distributions  
- Correlation heatmap[web:5][web:49]

---

## 🧠 Explainable AI (XAI)

SHAP was used to interpret model predictions at both global and local levels.

### Top Predictors

- Age  
- Hormonal Changes  
- Calcium Intake  
- Vitamin D Intake  
- Body Weight  
- Family History  

SHAP analysis confirmed that the model’s behavior aligns with biomedical expectations, with age, hormonal changes, and family history emerging as dominant contributors, consistent with prior ML and clinical studies on osteoporosis risk.[web:5][web:51]

---

## 🏥 CDSS Prototype

A prototype Clinical Decision Support System was implemented to:

- Receive patient-level inputs (e.g., age, hormonal changes, lifestyle factors)  
- Predict real-time osteoporosis risk using the trained Gradient Boosting model  
- Provide SHAP-based explanations for each prediction  
- Support clinicians in preventive decision-making and patient counseling[web:5]

Future versions may include web, mobile, or EHR integration for real-world clinical deployment.

---

## 🚧 Limitations

- Self-reported dataset rather than structured EHR data  
- Younger population than typical osteoporosis screening cohorts  
- No BMD (bone mineral density) or lab biomarkers included  
- Limited racial and ethnic diversity in some subgroups[web:49]

---

## 🔮 Future Work

- Integrate longitudinal clinical and imaging data (e.g., DXA, labs)  
- Add multimodal biomarkers to improve predictive performance  
- Deploy the CDSS as a web or mobile application  
- Perform fairness and subgroup performance analysis across demographic groups[web:5][web:51]

---

## 📁 Repository Structure

├── data/
│ ├── raw/
│ │ └── osteoporosis_lifestyle.csv
│ └── processed/
│ └── osteoporosis_clean.csv
├── notebooks/
│ ├── 01_eda.ipynb
│ ├── 02_preprocessing.ipynb
│ ├── 03_modeling.ipynb
│ └── 04_explainability_shap.ipynb
├── src/
│ ├── data_preprocessing.py
│ ├── train_models.py
│ ├── evaluate_models.py
│ ├── shap_analysis.py
│ └── cdss_app.py
├── models/
│ ├── gradient_boosting.joblib
│ └── scaler.joblib
├── reports/
│ ├── figures/
│ │ ├── roc_curves.png
│ │ ├── calibration_plot.png
│ │ ├── dca_plot.png
│ │ └── shap_summary.png
│ └── results_summary.md
├── requirements.txt
├── README.md
└── LICENSE


---

## 📚 References

1. Kanis JA, et al. European guidance for the diagnosis and management of osteoporosis.  
2. Warriner AH, Saag KG. Clinical risk assessment tools for osteoporosis.  
3. Vickers AJ, Elkin EB. Decision curve analysis for evaluating prediction models.  
4. Lundberg SM, Lee S-I. A unified approach to model interpretability (SHAP).  
5. Kaggle. Lifestyle Factors Influencing Osteoporosis. https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis[data][web:46][web:47]  
6. Additional machine learning and CDSS literature sources cited in the full project report.[web:5][web:51]

---

**Author:** Syllas Otutey  
**Program:** MS Health Informatics – Public Health Informatics & AI in Healthcare  
**Institution:** Michigan Technological University  

---

