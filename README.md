# Adult Census Income: EDA, Statistical Analysis, and Data Cleaning

An end-to-end exploratory and data-preparation project on the UCI **Adult (Census Income)** dataset. It examines how a person's income class (`<=50K` vs `>50K`) is associated with demographic, educational, employment, and financial characteristics, then produces a cleaned dataset for future modeling work.

> **Scope:** exploratory analysis, statistical testing, and data cleaning only. **No machine learning models are trained or evaluated in this repository.**

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange)
![Status](https://img.shields.io/badge/Scope-EDA%20%26%20Cleaning-informational)

---

## Table of Contents

- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Workflow](#workflow)
- [Key Results](#key-results)
- [Cleaned Dataset](#cleaned-dataset)
- [Getting Started](#getting-started)
- [Limitations and Notes](#limitations-and-notes)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Project Goals

1. Understand the structure, quality, and distributions of the Adult Census Income data.
2. Quantify how each feature relates to income using appropriate statistical tests, reporting **effect sizes alongside p-values**.
3. Identify data-quality problems and resolve them through documented, justified decisions.
4. Export a clean, validated dataset while keeping the raw data untouched.

---

## Dataset

| Item | Detail |
|---|---|
| Source | [UCI Machine Learning Repository: Adult](https://archive.ics.uci.edu/dataset/2/adult) (ID 2) |
| Access | Loaded programmatically with `ucimlrepo` (`fetch_ucirepo(id=2)`) |
| Origin | 1994 U.S. Census database, extracted by Barry Becker |
| Size | 48,842 records, 14 features + 1 target |
| Target | `income`: annual income `<=50K` or `>50K` |
| DOI | 10.24432/C5XW20 |

**Features:** 6 numerical (`age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`) and 8 categorical (`workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`).

`fnlwgt` is a census **sampling weight**, not an income measure. `education-num` is an ordinal encoding of `education`.

---

## Repository Structure

```
adult-census-data-analysis/
│
├── data/
│   ├── raw/                 # Reserved for raw data (notebooks fetch it directly from UCI)
│   └── processed/           # Cleaned dataset (adult_income_cleaned.csv)
│
├── notebooks/
│   ├── EDA_and_Statistical_Analysis.ipynb
│   └── Data_Cleaning_and_Preprocessing.ipynb
│
├── reports/
│   └── EDA_Report.md        # Written report of the EDA and statistical findings
│
├── README.md
└── requirements.txt
```

---

## Workflow

### 1. EDA and Statistical Analysis (`.ipynb`)

- Data quality assessment, including the target label problem (see below).
- Univariate analysis of all numerical and categorical features.
- Bivariate analysis of every feature against income.
- Statistical testing:
  - **Mann–Whitney U** with **rank-biserial correlation** for numerical features.
  - **Chi-square test of independence** with **Cramér's V** for categorical features, including expected-count checks.
  - A **two-part analysis** for zero-inflated `capital-gain` and `capital-loss` (any positive value, then amount among positives).

The full write-up is in [`reports/EDA_Report.md`](reports/EDA_Report.md).

### 2. Data Cleaning and Preprocessing (`.ipynb`)

The raw data is preserved in `df_raw`; all cleaning is done on a working copy.

| Step | Action | Result |
|---|---|---|
| Target standardization | Removed trailing periods from labels (`<=50K.` → `<=50K`, `>50K.` → `>50K`) | 4 raw labels → 2 classes |
| Missing value standardization | Converted the `"?"` marker to `NaN` in `workclass`, `occupation`, `native-country` | `NaN` and `"?"` never overlapped, so they were safely merged |
| Duplicate handling | Removed exact duplicates on features + standardized target; kept records with identical features but conflicting labels | 53 rows removed in total (see note below) |
| Missing value imputation | Mode imputation for the three affected categorical columns | `Private`, `Prof-specialty`, `United-States` |
| Validation | Checked missing values, dtypes, duplicates, and target values | 0 missing values, 0 remaining duplicates |
| Export | Dropped the raw `income` column and renamed `income_clean` to `income` | `adult_income_cleaned.csv` |

**Why the target needed cleaning:** the raw `income` column held four values (`<=50K`, `<=50K.`, `>50K`, `>50K.`). The trailing period reflects how the training and test subsets are labelled; it does not denote a different class.

**Why columns were not dropped:** chi-square tests on the affected features showed association with income (χ² = 1,609.50 for `workclass`, 5,975.47 for `occupation`, 449.41 for `native-country`), so the notebook kept them and imputed instead.

> **Duplicate counts:** 29 exact duplicates exist under the raw labels, rising to 52 once labels are standardized. The first pass removed 52 rows (48,842 → 48,790). A second pass after imputation removed 1 more (→ 48,789), most likely because imputation made one additional pair of rows identical (the notebook does not state the cause). Five feature combinations (10 records) share identical features but have opposite labels and were deliberately retained.

---

## Key Results

### Data quality

| Issue | Finding |
|---|---|
| Target labels | 4 variants of 2 classes, fixed |
| Missing data | Two representations (`NaN` and `"?"`): `workclass` 2,799 (5.73%), `occupation` 2,809 (5.75%), `native-country` 857 (1.75%) |
| Co-missingness | After de-duplication, all 2,795 missing `workclass` rows also lack `occupation`; 10 more rows lack only `occupation` |
| Class imbalance | `<=50K` 76.07% / `>50K` 23.93% (raw); 76.06% / 23.94% (cleaned) |
| Skewed or sparse features | `capital-gain`, `capital-loss`, `fnlwgt` |

### Strongest associations with income (raw data, 48,842 records)

| Feature | Test | Effect size | Reading |
|---|---|---|---|
| `relationship` | Chi-square | Cramér's V = 0.4545 | Medium to large |
| `marital-status` | Chi-square | V = 0.4483 | Medium to large |
| `education-num` | Mann–Whitney U | Rank-biserial *r* = 0.4325 | Medium to large |
| `education` | Chi-square | V = 0.3659 | Medium |
| `age` | Mann–Whitney U | *r* = 0.3645 | Medium |
| `occupation` | Chi-square | V = 0.3501 | Medium |
| `hours-per-week` | Mann–Whitney U | *r* = 0.3437 | Medium |
| `capital-gain` (any positive) | Chi-square | V = 0.2658 | Small to medium |
| `sex` | Chi-square | V = 0.2146 | Small to medium |
| `race`, `native-country` | Chi-square | V ≈ 0.10 | Weak |
| `fnlwgt` | Mann–Whitney U | *r* = −0.0080 (*p* = 0.19) | Negligible |

Almost every test is statistically significant because of the large sample, so **effect sizes, not p-values, drive the interpretation**. Effect-size labels are conventional rules of thumb. For per-group statistics, assumptions, and interpretation, see the [full report](reports/EDA_Report.md).

All findings are **associations, not causal effects**: they are unadjusted, one-feature-at-a-time comparisons, and several features overlap (for example, `relationship` and `marital-status`).

---

## Cleaned Dataset

`adult_income_cleaned.csv` has **48,789 rows × 15 columns** (14 features + `income`).

| Metric | Value |
|---|---:|
| Original records | 48,842 |
| Final records | 48,789 |
| `<=50K` | 37,108 (76.06%) |
| `>50K` | 11,681 (23.94%) |
| Missing values | 0 |
| Remaining exact duplicates | 0 |

The notebook performs cleaning only. It does **not** apply categorical encoding, scaling, outlier treatment, or feature engineering.

---

## Getting Started

### Prerequisites

- Python 3.x
- Internet access (the dataset is downloaded from UCI at run time)

### Installation

```bash
git clone https://github.com/Omnia-Elsaka10/adult-census-data-analysis.git
cd adult-census-data-analysis

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Run the notebooks

```bash
jupyter notebook
```

Open the notebooks in order:

1. `notebooks/EDA_and_Statistical_Analysis.ipynb`
2. `notebooks/Data_Cleaning_and_Preprocessing.ipynb`

**Running notebook 02 outside Google Colab:**

- The final code cell uses `google.colab.files.download`, which only works in Colab. Skip or remove it locally.
- The CSV is written to the notebook's working directory. To save it directly into the repository, set `output_path = "../data/processed/adult_income_cleaned.csv"` before exporting.
- The first cell of each notebook runs `pip install ucimlrepo`; this is already covered by `requirements.txt`.

---

## Limitations and Notes

- **Mode imputation can distort relationships.** Replacing missing values with the most frequent category reduces variability, and the imputed values are not real observations. In the cleaning notebook's own output, records with missing `workclass` or `occupation` have a `>50K` rate of about 9.5%, whereas the imputed categories have much higher rates (`Private` 21.80%, `Prof-specialty` 45.13%). Imputing these rows may therefore weaken or shift the association with income. An explicit `Unknown` category or a missingness indicator would be worth comparing in future work.
- **Possible leakage in future modeling.** Imputation values were computed on the full dataset. For predictive modeling, learned preprocessing parameters should be fitted on training data only.
- **Class imbalance (about 3:1).** Evaluation metrics for any future classifier should account for it.
- **Sampling weights are not used.** `fnlwgt` is retained as a feature, and all proportions and tests are unweighted, so results describe this sample rather than the 1994 U.S. population.
- **Sparse categories.** The `native-country` chi-square test violates the expected-count assumption (two cells with expected count below 1), so its p-value and effect size should be read cautiously.
- **Conflicting duplicates retained.** Ten records share identical features but have opposite labels, which implies some irreducible label noise.
- **Historical data.** The data come from 1994, and income is dichotomised at a single nominal-dollar threshold.
- **Association, not causation.** Observed differences, including those by `sex`, `race`, and `native-country`, are unadjusted and should not be interpreted as causal effects.

---

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · SciPy (`mannwhitneyu`, `chi2_contingency`) · ucimlrepo · Jupyter

---

## References

- Becker, B., & Kohavi, R. *Adult* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20
