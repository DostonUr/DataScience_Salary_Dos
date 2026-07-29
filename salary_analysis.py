"""
Data Scientist Salary Prediction
--------------------------------
Cleans a public, self-reported salary dataset (2019-2021) for data
science / analytics roles and compares a Linear Regression baseline
against a Random Forest Regressor.

Data source:
https://raw.githubusercontent.com/archiewood/ds_salaries/refs/heads/master/final_salary_data.csv

Author: Doston Urinov
https://github.com/DostonUr | https://www.linkedin.com/in/doston-urinov/
"""

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error

DATA_URL = (
    "https://raw.githubusercontent.com/archiewood/ds_salaries/"
    "refs/heads/master/final_salary_data.csv"
)


def group_title(title: str) -> str:
    """Collapse ~115 raw free-text job titles into 8 seniority buckets."""
    if pd.isna(title):
        return "other"
    t = title.lower()
    if "director" in t or "head of" in t or "vp" in t or "principal" in t:
        return "leadership"
    if "manager" in t or "lead" in t:
        return "manager/lead"
    if "senior" in t or "sr" in t or "senor" in t:  # covers a real typo in the data
        return "senior ic"
    if "junior" in t or "associate" in t or "entry" in t:
        return "junior ic"
    if "analyst" in t:
        return "analyst"
    if "machine learning" in t or "ml engineer" in t or "mlops" in t:
        return "ml engineer"
    if "scientist" in t:
        return "data scientist (mid)"
    return "other"


def load_and_clean(url: str = DATA_URL) -> pd.DataFrame:
    df = pd.read_csv(url)
    df["title_group"] = df["title"].apply(group_title)
    df = df.dropna(subset=["salary_usd"]).copy()
    df["tenure_clean"] = df["tenure_clean"].fillna(df["tenure_clean"].median())
    df["education_level"] = df["education_level"].fillna("Unspecified")
    df["industry_group"] = df["industry_group"].fillna("Other industry")
    return df


def build_pipeline(model):
    cat_cols = ["title_group", "us_region", "education_level", "industry_group"]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", model)])


def main():
    df = load_and_clean()
    features = ["title_group", "tenure_clean", "us_region", "education_level", "industry_group"]
    X, y = df[features], df["salary_usd"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear = build_pipeline(LinearRegression())
    linear.fit(X_train, y_train)
    pred_linear = linear.predict(X_test)
    print(f"Linear  R2: {r2_score(y_test, pred_linear):.3f}  "
          f"MAE: ${mean_absolute_error(y_test, pred_linear):,.0f}")

    rf = build_pipeline(RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42))
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    print(f"RandomForest  R2: {r2_score(y_test, pred_rf):.3f}  "
          f"MAE: ${mean_absolute_error(y_test, pred_rf):,.0f}")

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    print(f"RandomForest 5-fold CV R2: {cv_scores.round(3)} (mean {cv_scores.mean():.3f})")

    # Feature importance, grouped back to original category (not one-hot columns)
    rf.fit(X, y)
    ohe = rf.named_steps["pre"].named_transformers_["cat"]
    cat_cols = ["title_group", "us_region", "education_level", "industry_group"]
    feat_names = list(ohe.get_feature_names_out(cat_cols)) + ["tenure_clean"]
    importances = rf.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
    imp_df["base"] = imp_df["feature"].apply(
        lambda f: next((c for c in cat_cols if f.startswith(c + "_")), f)
    )
    print("\nGrouped feature importance:")
    print(imp_df.groupby("base")["importance"].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
