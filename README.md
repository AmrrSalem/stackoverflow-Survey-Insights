# 🧠 Developer Salary Prediction Using Stack Overflow Survey 2024

This repository presents a machine learning pipeline for predicting developer salaries using the 2024 Stack Overflow Developer Survey. It uses a Random Forest model and SHAP values to explain feature importance.

---

## 📌 Motivation

The aim of this project is to:
- Predict developer salaries using experience, education, age, and country.
- Identify the most impactful features that influence compensation.
- Provide interpretable results using SHAP values.


---

## 📂 Repository Structure

| File Name                                     | Description                                                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `rf_salary_model_notebook_with_answers.ipynb` | Main Jupyter notebook with all code, five business questions, and written answers interpreting the results.          |
| `rf_full_features_engineered.py`              | Original Python script for data loading, feature engineering, Random Forest modeling, and SHAP-based interpretation. |
| `Readme.md`                                   | Contains an introduction to the findings, links to the data used, and the packages used. |


---

## 🛠️ Libraries Used

- `pandas`, `numpy` — Data manipulation
- `scikit-learn` — Data preprocessing, modeling, and evaluation
- `shap` — Model explainability
- `matplotlib`/`seaborn` (optional for visualizations)
- `logging` — Status reporting

---

## 📈 Summary of Results

- The Random Forest model achieves solid R² scores in cross-validation and test evaluation.
- SHAP analysis indicates that **Country**, **Years of Professional Coding Experience**, and **Education Level** are the top predictors of salary.
- The notebook answers 5 business questions related to salary prediction and feature importance.

---

## 🙏 Acknowledgements

- Data is sourced from the [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co/datasets/stack-overflow-developer-survey-2024.zip).
- Thanks to the open-source community for tools like `scikit-learn` and `shap`.
- Assisted by [ChatGPT](https://openai.com/chatgpt) for code suggestions and debugging support.


