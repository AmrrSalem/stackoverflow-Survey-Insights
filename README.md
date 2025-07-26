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
## 🧰 Python Libraries and Tools


| Library        | Purpose                                                         | Official Source                                                        |
| -------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `pandas`       | Data loading, filtering, and manipulation                       | [https://pandas.pydata.org/](https://pandas.pydata.org/)               |
| `numpy`        | Numeric operations and handling missing values                  | [https://numpy.org/](https://numpy.org/)                               |
| `scikit-learn` | Machine learning pipeline, preprocessing, modeling, and metrics | [https://scikit-learn.org/](https://scikit-learn.org/)                 |
| `shap`         | Model interpretability (SHAP values for feature importance)     | [https://github.com/slundberg/shap](https://github.com/slundberg/shap) |
| `logging`      | Informational logs for process tracking                         | Python standard library                                                |

---
## <img width="20" height="20" alt="image" src="https://github.com/user-attachments/assets/53e78f24-8cdd-43c2-9324-a42b9687468f" />  Install Dependencies 
Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt

---

---

## 📈 Summary of Results

- The Random Forest model achieves solid R² = {0.34} scores in cross-validation and test evaluation.
- SHAP analysis indicates that **Country**, **Years of Professional Coding Experience**, and **Education Level** are the top predictors of salary.
- The notebook answers 5 business questions related to salary prediction and feature importance.
- Medium's blog post provides a concise version of the results and findings. https://medium.com/@amrr.salem/developer-salaries-why-location-matters-more-than-you-think-cf38b6796014

---

## 🙏 Acknowledgements

- Data is sourced from the [Stack Overflow Developer Survey 2024](https://survey.stackoverflow.co/datasets/stack-overflow-developer-survey-2024.zip).
- Thanks to the open-source community for tools like `scikit-learn` and `shap`.
---
## 🔥 Inspirational Repositories
The following open-source projects served as inspiration and reference points for this work:

tien02/salary-prediction

recodehive/Stackoverflow-Analysis


