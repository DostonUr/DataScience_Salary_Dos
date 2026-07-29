# Data Scientist Salary Prediction

Cleans and models a public, self-reported salary dataset (2019–2021, 258 rows) for
data science / analytics roles, and compares a Linear Regression baseline against
a Random Forest Regressor.

Full write-up with commentary on the modeling decisions and their limits:
**[Read the article](https://urinovd.github.io/2026/08/15/data-scientist-salary-prediction.html)**

## What's here

- `salary_analysis.py` — end-to-end cleaning and modeling script
- `requirements.txt` — dependencies

## Data

Source: [archiewood/ds_salaries](https://github.com/archiewood/ds_salaries),
`final_salary_data.csv` — self-reported salary data for data science/analytics
roles, mostly US-based, 2019–2021.

## Key steps

1. **Title normalization** — 115 raw free-text job titles collapsed into 8
   seniority buckets (junior IC, analyst, mid-level data scientist, ML engineer,
   senior IC, manager/lead, leadership, other) via keyword matching.
2. **Missing-value handling** — tenure filled with median, education/industry
   filled with an explicit "Unspecified"/"Other" category rather than dropped.
3. **Outlier review** — three salaries above $370k were checked individually
   (plausible senior/staff West Coast roles) and kept rather than removed.
4. **Modeling** — Linear Regression baseline vs. Random Forest Regressor,
   compared on held-out test data and 5-fold cross-validation.

## Results

| Model | R² (test) | MAE (test) |
|---|---|---|
| Linear Regression | 0.27 | ~$41,600 |
| Random Forest | 0.24 | ~$43,300 |

Random Forest did not outperform the linear baseline — with only 254 usable
rows, the added flexibility mostly increased variance rather than capturing
real signal. 5-fold CV R² ranged from 0.10 to 0.34 depending on the split,
which is itself informative about how much this result should be trusted.

## Run it

```bash
pip install -r requirements.txt
python salary_analysis.py
```

## Author

Doston Urinov — [LinkedIn](https://www.linkedin.com/in/doston-urinov/) ·
[Kaggle](https://www.kaggle.com/dostonur) · [GitHub](https://github.com/DostonUr)
