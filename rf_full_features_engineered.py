import numpy as np
import pandas as pd
import shap
import logging
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


RANDOM_SEED = 42
N_ESTIMATORS_RF = 200
SHAP_BACKGROUND_SAMPLES = 50
SHAP_TEST_SAMPLES = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_years_experience(years_str):
    if pd.isna(years_str): return np.nan
    if years_str == 'Less than 1 year': return 0.5
    if years_str == 'More than 50 years': return 51.0
    try:
        return float(years_str)
    except:
        return np.nan


def parse_age(age_str):
    age_map = {
        "Under 18 years old": 16.0,
        "18-24 years old": 21.0,
        "25-34 years old": 29.5,
        "35-44 years old": 39.5,
        "45-54 years old": 49.5,
        "55-64 years old": 59.5,
        "65 years or older": 70.0,
    }
    return age_map.get(age_str, np.nan)





def load_and_prepare_data(
        public_csv_path: str = 'survey_results_public.csv',
        random_seed: int = RANDOM_SEED
):
    """
    Load the public survey results, prepare features, and split into train/test.

    Parameters
    ----------
    public_csv_path : str
        Path to the extracted 'survey_results_public.csv' file.
    random_seed : int
        Seed for reproducible train/test split.

    Returns
    -------
    (X_train, X_test, y_train, y_test), num_cols, cat_cols
        - X_train, X_test: DataFrames of numeric + categorical features
        - y_train, y_test: Series of log-salary target
        - num_cols: list of numeric column names
        - cat_cols: list of categorical column names
    """
    logging.info(f"Loading {public_csv_path}...")
    df = pd.read_csv(public_csv_path)

    # Filter out missing salaries and engineer new features
    df = df[df['ConvertedCompYearly'].notna()].copy()
    df['LogSalary'] = np.log1p(df['ConvertedCompYearly'])
    df['YearsCodeProNum'] = df['YearsCodePro'].apply(parse_years_experience)
    df['AgeNum'] = df['Age'].apply(parse_age)

    # Final feature lists
    num_cols = ['YearsCodeProNum', 'AgeNum']
    cat_cols = ['EdLevel', 'Country']

    # Assemble X and y, then split
    X = df[num_cols + cat_cols]
    y = df['LogSalary']
    split = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    return split, num_cols, cat_cols


def build_pipeline(num_cols, cat_cols):
    num_pipe = Pipeline([
        ('impute', KNNImputer(n_neighbors=5)),
        ('scale', StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    rf = RandomForestRegressor(n_estimators=N_ESTIMATORS_RF,
                               random_state=RANDOM_SEED,
                               n_jobs=-1)
    return Pipeline([('preproc', preprocessor), ('rf', rf)])


def evaluate_model(model, X_train, y_train, X_test, y_test):
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    print("\n=== Cross-Validated R² Scores ===")
    for i, score in enumerate(scores, 1):
        print(f"Fold {i}: R² = {score:.4f}")
    print(f"Mean R² : {np.mean(scores):.4f}")
    print(f"Std Dev : {np.std(scores):.4f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("\n=== Test Set Evaluation ===")
    print(f"R²   : {r2:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print("============================")


def shap_summary(pipeline, X_train, X_test, num_samples=SHAP_TEST_SAMPLES):
    logging.info("Computing SHAP values...")
    preproc = pipeline.named_steps['preproc']
    model = pipeline.named_steps['rf']

    X_train_trans = preproc.transform(X_train)
    X_test_subset = X_test.iloc[:num_samples]
    X_test_trans = preproc.transform(X_test_subset)
    background = shap.sample(X_train_trans, SHAP_BACKGROUND_SAMPLES, random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(model, data=background)
    shap_vals = explainer(X_test_trans, check_additivity=False).values

    feature_names = preproc.get_feature_names_out()
    abs_shap = np.abs(shap_vals)
    mean_abs = np.mean(abs_shap, axis=0)
    df_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs
    }).sort_values('mean_abs_shap', ascending=False)

    print("\n--- SHAP Feature Importance (Top 20) ---")
    print(df_shap.head(20).to_string(index=False))
    print("----------------------------------------")


def main():
    (X_train, X_test, y_train, y_test), num_cols, cat_cols = load_and_prepare_data()
    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)
    evaluate_model(pipeline, X_train, y_train, X_test, y_test)
    shap_summary(pipeline, X_train, X_test)


if __name__ == '__main__':
    main()
