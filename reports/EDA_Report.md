# Adult Census Income: Exploratory Data Analysis and Statistical Analysis

## 1. Project Overview

This project presents an exploratory data analysis (EDA) and statistical investigation of the **Adult Census Income dataset** from the UCI Machine Learning Repository.

The dataset contains demographic, educational, employment, and financial information. Its original classification task is to determine whether an individual's annual income exceeds $50,000.

The objective of this project is not to build a predictive model, but to develop a thorough understanding of the dataset before any future machine learning work.

The analysis focuses on:

- Understanding the dataset structure, feature distributions, and data quality.
- Identifying inconsistencies and missing-value patterns.
- Exploring numerical and categorical features individually.
- Investigating how each feature relates to the income category.
- Applying statistical tests and effect-size measures to quantify observed differences.
- Identifying potential limitations and considerations for subsequent data preparation.

**Scope:** This project covers exploratory and inferential analysis only. No machine learning models were trained or evaluated.

---

## 2. Dataset Description

### 2.1 Data Source

The dataset was obtained from the **UCI Machine Learning Repository** using the `ucimlrepo` package.

| Property | Description |
|---|---|
| Dataset | Adult (Census Income) |
| Repository | UCI Machine Learning Repository |
| Dataset ID | 2 |
| DOI | 10.24432/C5XW20 |
| Original source | 1994 U.S. Census database |
| Creators | Barry Becker and Ronny Kohavi |
| Records | 48,842 |
| Original columns | 15 |

**Dataset link:** https://archive.ics.uci.edu/dataset/2/adult

### 2.2 Features

The dataset contains 14 input features and one target variable.

| Feature Type | Features |
|---|---|
| Numerical (6) | `age`, `fnlwgt`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week` |
| Categorical (8) | `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country` |

An important observation is that `education` and `education-num` represent the same educational information using categorical and numerical encodings, respectively.

Additionally, `fnlwgt` represents a census sampling weight rather than an individual's income or wealth.

### 2.3 Target Variable

The target variable is `income`, which distinguishes between two income categories:

- `<=50K`: Annual income is at most $50,000.
- `>50K`: Annual income exceeds $50,000.

The original target distribution was inconsistent because the same classes appeared with and without trailing periods. This issue was addressed during the initial exploration, as described in the next section.

---

## 3. Data Quality Assessment

### 3.1 Target Variable Inconsistency

The original `income` column contained four distinct labels instead of the expected two:

| Original Label | Count |
|---|---:|
| `<=50K` | 24,720 |
| `<=50K.` | 12,435 |
| `>50K` | 7,841 |
| `>50K.` | 3,846 |

The trailing period represented a formatting inconsistency rather than a different income class.

A cleaned target column, `income_clean`, was created by removing surrounding whitespace and trailing periods.

The resulting distribution was:

| Income Category | Count | Percentage |
|---|---:|---:|
| `<=50K` | 37,155 | 76.07% |
| `>50K` | 11,687 | 23.93% |
| **Total** | **48,842** | **100%** |

The dataset therefore exhibits a noticeable class imbalance, with approximately 3.2 observations in the lower-income class for every observation in the higher-income class.

### 3.2 Missing Values and Placeholder Categories

The initial inspection revealed two different representations of missing information:

1. Actual missing values (`NaN`).
2. The literal placeholder `?`.

Both appeared in the following categorical features:

| Feature | NaN | `?` | Combined Missing / Placeholder |
|---|---:|---:|---:|
| `workclass` | 963 | 1,836 | 2,799 |
| `occupation` | 966 | 1,843 | 2,809 |
| `native-country` | 274 | 583 | 857 |

The combined counts represent approximately 5.7% of the dataset for `workclass`, 5.8% for `occupation`, and 1.8% for `native-country`.

An important distinction is that pandas recognizes `NaN` as missing data, whereas `?` is treated as an ordinary categorical value unless explicitly handled.

During the visual and statistical analysis, actual missing values were represented as `Missing`, while `?` remained a separate category. Consequently, these two forms of unknown information were analyzed separately.

### 3.3 Numerical Distribution Characteristics

The numerical features exhibited different distribution patterns:

| Feature | Key Observations |
|---|---|
| `age` | Ranges from 17 to 90, with a right-skewed distribution. |
| `hours-per-week` | Ranges from 1 to 99 and is strongly concentrated around 40 hours. |
| `fnlwgt` | Has a wide range and several upper-end outliers; its distribution appears more symmetric on a logarithmic scale. |
| `capital-gain` | Highly zero-inflated, with the first three quartiles equal to zero and a noticeable spike at 99,999. |
| `capital-loss` | Also zero-inflated, with more than 75% of observations equal to zero. |

Both capital-related features contain more than 40,000 zero-valued observations. Their means are therefore strongly influenced by a relatively small number of positive values.

This motivated a two-part analytical approach: examining whether a positive value exists, followed by analyzing the magnitude among observations with positive values only.

### 3.4 Rare Categories

Several categorical features contain categories with very few observations:

- `workclass`: `Never-worked` (10) and `Without-pay` (21).
- `occupation`: `Armed-Forces` (15).
- `marital-status`: `Married-AF-spouse` (37).
- `native-country`: Several countries have very small representation.

These low-frequency categories are particularly relevant when interpreting proportions and applying statistical tests.

---

## 4. Exploratory Data Analysis

### 4.1 Univariate Analysis

Univariate analysis was performed to understand the distribution of each feature independently.

#### Categorical Features

Several categories dominate the dataset:

- **Workclass:** `Private` is the most common employment category, representing approximately 69.4% of the observations.
- **Education:** `HS-grad` is the most frequent category (15,784 observations), followed by `Some-college` (10,878) and `Bachelors` (8,025).
- **Marital Status:** `Married-civ-spouse` (22,379) and `Never-married` (16,117) account for most observations.
- **Occupation:** `Prof-specialty`, `Craft-repair`, and `Exec-managerial` are among the largest categories.
- **Relationship:** `Husband` (19,716) and `Not-in-family` (12,583) are the most frequent categories.
- **Race:** `White` represents approximately 85.5% of the dataset, followed by `Black` at approximately 9.6%.
- **Sex:** Approximately 66.9% of observations are male and 33.1% are female.
- **Native Country:** `United-States` accounts for approximately 90% of non-missing observations, with `Mexico` being the largest foreign-country category.

#### Numerical Features

The following table summarizes the numerical distributions:

| Feature | Mean | Std | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `age` | 38.64 | 13.71 | 17 | 28 | 37 | 48 | 90 |
| `education-num` | 10.08 | 2.57 | 1 | 9 | 10 | 12 | 16 |
| `hours-per-week` | 40.42 | 12.39 | 1 | 40 | 40 | 45 | 99 |
| `fnlwgt` | 189,664 | 105,604 | 12,285 | 117,550 | 178,144 | 237,642 | 1,490,400 |
| `capital-gain` | 1,079 | 7,452 | 0 | 0 | 0 | 0 | 99,999 |
| `capital-loss` | 87.5 | 403.0 | 0 | 0 | 0 | 0 | 4,356 |

The numerical distributions highlight several important characteristics:

- The age distribution is right-skewed, with most observations concentrated among younger adults.
- `hours-per-week` is strongly concentrated around the standard 40-hour working week.
- `fnlwgt` has a wide range and substantial variability.
- Both capital-related variables are highly zero-inflated, making their means less representative of a typical observation.

### 4.2 Bivariate Analysis: Numerical Features vs. Income

Numerical features were compared across the two income categories to identify differences in their distributions.

| Feature | `<=50K` Median | `>50K` Median | Main Observation |
|---|---:|---:|---|
| `age` | 34 | 43 | The higher-income group tends to be older. |
| `education-num` | 9 | 12 | The higher-income group tends to have higher educational levels. |
| `hours-per-week` | 40 | 40 | Both groups share the same median, but the higher-income group has a higher upper quartile. |
| `fnlwgt` | 178,811 | 176,729 | The two distributions are very similar. |

For `hours-per-week`, both groups have a median of 40 hours. However, the mean is approximately 45.5 hours for the higher-income group compared with 38.8 hours for the lower-income group. The upper quartile is also higher (50 versus 40).

This demonstrates why examining the full distribution is important rather than relying exclusively on the median.

#### Capital Gain and Capital Loss

The capital-related features were investigated using two perspectives:

1. Whether an observation has a zero or positive value.
2. The distribution of positive values only.

The analysis showed that:

- Approximately 21% of the higher-income group has positive capital gains, compared with approximately 4% of the lower-income group.
- Positive capital gains are substantially larger in the higher-income group.
- Positive capital losses are also more common in the higher-income group, although the difference is less pronounced.
- Among observations with positive capital losses, the two groups have similar medians.

These findings indicate that both the occurrence and magnitude of capital-related values can differ across income categories. However, positive-only comparisons apply exclusively to the subset with non-zero values.

### 4.3 Bivariate Analysis: Categorical Features vs. Income

Categorical features were examined using income proportions within each category.

Some of the main observations include:

| Feature | Categories with Higher `>50K` Proportions |
|---|---|
| `workclass` | `Self-emp-inc` (>55%), `Federal-gov` (approximately 40%) |
| `education` | `Prof-school` and `Doctorate` (>70%), `Masters` (approximately 55%) |
| `marital-status` | `Married-civ-spouse` (approximately 45%) |
| `occupation` | `Exec-managerial` and `Prof-specialty` (approximately 45% or more) |
| `relationship` | `Husband` (approximately 45%), `Wife` (approximately 47%) |
| `sex` | Male (approximately 30%) |
| `race` | `Asian-Pac-Islander` (approximately 27%), `White` (approximately 25%) |
| `native-country` | Some small categories, including `France`, `India`, `Taiwan`, and `Iran`, show relatively high proportions. |

A particularly important observation concerns `native-country`: some categories with high income proportions contain very few observations. Their percentages are therefore unstable and should not be interpreted as reliable comparisons between countries.

---

## 5. Statistical Analysis

Visual exploration helps identify patterns, but statistical tests provide a more formal way to investigate whether observed differences are supported by the sample.

The analysis uses statistical significance together with effect sizes. This distinction is important because a large sample can produce very small p-values even when the practical difference is limited.

### 5.1 Statistical Methods

| Analysis | Statistical Test | Effect Size |
|---|---|---|
| Numerical features vs. income | Mann–Whitney U test | Rank-biserial correlation |
| Categorical features vs. income | Chi-square test of independence | Cramér's V |
| Capital gain/loss occurrence | Chi-square test of independence | Cramér's V |
| Positive capital gain/loss values | Mann–Whitney U test | Rank-biserial correlation |

#### Mann–Whitney U Test

The Mann–Whitney U test was used to compare the distributions of numerical features between the two independent income groups.

It is a rank-based non-parametric test and does not require the data to follow a normal distribution.

The rank-biserial correlation was calculated as:

\[
r_{rb} = \frac{2U}{n_1 n_2} - 1
\]

where group 1 represents observations with income `>50K`.

A positive value indicates that observations in the higher-income group tend to have larger values for the feature.

The test evaluates differences in distributions and stochastic ordering; it should not automatically be interpreted as a test of medians.

#### Chi-Square Test of Independence

The Chi-square test was used to investigate whether categorical features are statistically associated with the income category.

Cramér's V was calculated to measure the strength of these associations:

\[
V = \sqrt{\frac{\chi^2}{n \times \min(r-1,c-1)}}
\]

where:

- \(\chi^2\) is the Chi-square statistic.
- \(n\) is the total number of observations.
- \(r\) is the number of rows.
- \(c\) is the number of columns.

For interpretation, conventional rules of thumb were used:

| Effect Size | Approximate Cramér's V |
|---|---:|
| Small | 0.10 |
| Medium | 0.30 |
| Large | 0.50 |

These thresholds are guidelines rather than strict boundaries.

### 5.2 Numerical Features: Mann–Whitney U Results

| Feature | U Statistic | p-value | Rank-biserial r |
|---|---:|---:|---:|
| `age` | 296,257,069.5 | Extremely small | 0.3645 |
| `hours-per-week` | 291,741,763.0 | Extremely small | 0.3437 |
| `education-num` | 311,010,711.5 | Extremely small | 0.4325 |
| `fnlwgt` | 215,371,681.5 | 0.1897 | -0.0080 |
| `capital-gain` (positive only) | 3,626,235.5 | Extremely small | 0.8857 |
| `capital-loss` (positive only) | 901,355.5 | 2.76 × 10⁻⁵⁷ | 0.3847 |

**Key interpretations:**

- `education-num` shows the strongest association among the main numerical features, with a rank-biserial correlation of 0.4325.
- `age` and `hours-per-week` also show moderate distributional differences between income groups.
- The `fnlwgt` result is not statistically significant (p = 0.1897), and its effect size is close to zero.
- Positive `capital-gain` values show a very strong difference between the groups. However, this result applies only to observations with positive capital gains.
- Positive `capital-loss` values also differ between groups, although the effect is less pronounced.

**Important p-value note:** Some p-values were displayed as `0.00000e+00` because of floating-point underflow during computation. They should not be interpreted as exactly zero. Instead, they indicate extremely small p-values.

With a dataset of this size, statistical significance alone is insufficient to judge the importance of a relationship. Effect sizes provide additional information about the magnitude of observed differences.

### 5.3 Categorical Features: Chi-Square Results

| Feature | Chi-Square Statistic | Degrees of Freedom | p-value | Cramér's V |
|---|---:|---:|---|---:|
| `relationship` | 10,088.72 | 5 | Extremely small | 0.4545 |
| `marital-status` | 9,816.02 | 6 | Extremely small | 0.4483 |
| `education` | 6,537.97 | 15 | Extremely small | 0.3659 |
| `occupation` | 5,985.71 | 15 | Extremely small | 0.3501 |
| `capital-gain` (any positive) | 3,451.51 | 1 | Extremely small | 0.2658 |
| `sex` | 2,248.85 | 1 | Extremely small | 0.2146 |
| `workclass` | 1,613.32 | 9* | Extremely small | 0.1817 |
| `capital-loss` (any positive) | 901.46 | 1 | 4.72 × 10⁻¹⁹⁸ | 0.1359 |
| `race` | 487.03 | 4 | 4.28 × 10⁻¹⁰⁴ | 0.0999 |
| `native-country` | 452.62 | 42 | 2.90 × 10⁻⁷⁰ | 0.0963 |

*The degrees of freedom for `workclass` are derived from its contingency table dimensions.*

**Key interpretations:**

- `relationship` and `marital-status` show the strongest categorical associations with income.
- `education` and `occupation` also demonstrate meaningful associations.
- `sex` and `workclass` show smaller but noticeable associations.
- `race` and `native-country` have relatively weak effect sizes.
- The capital-gain occurrence indicator has a stronger association than the capital-loss occurrence indicator.

#### Expected-Frequency Assumptions

The expected-frequency condition was also examined when applying the Chi-square test.

The `native-country` contingency table contains expected frequencies below 1, meaning the usual expected-count assumptions are not satisfied. Its statistical results should therefore be interpreted cautiously.

Some low expected frequencies were also observed in `occupation` and `workclass`, although the proportions of affected cells were smaller.

These checks are important because a statistically significant p-value does not automatically guarantee that the assumptions of the test are satisfied.

### 5.4 Association vs. Causation

The statistical results describe associations between individual features and income. They do not establish causal relationships.

For example:

- The association between education and income does not prove that education alone causes higher income.
- The relationship and marital-status features overlap structurally, so their strong associations should not be considered independent evidence.
- Age may be related to education, employment experience, marital status, and capital-related features.
- The observed difference between male and female income proportions is unadjusted for other variables such as occupation, working hours, and marital status.

A multivariable analysis would be required to investigate the independent contribution of these features.

---

## 6. Key Findings

The analysis produced several important findings:

1. **Target inconsistency:** The original target contained four formatting variations, which were standardized into two income classes. The resulting distribution is approximately 76% `<=50K` and 24% `>50K`.

2. **Household and marital characteristics:** `relationship` and `marital-status` show the strongest categorical associations with income, with Cramér's V values of 0.4545 and 0.4483, respectively.

3. **Education:** Educational attainment demonstrates a consistent association with income. `education-num` has a rank-biserial correlation of 0.4325, while `education` has a Cramér's V of 0.3659.

4. **Age and working hours:** The higher-income group is older on average and tends to work longer hours. However, both groups have the same median working hours of 40.

5. **Employment characteristics:** Occupation shows a meaningful association with income (V = 0.3501), while workclass has a smaller association (V = 0.1817).

6. **Capital gain and loss:** Positive capital gains are considerably more common in the higher-income group, and their positive values show a particularly strong distributional difference. Capital-loss patterns are also associated with income, but less strongly.

7. **Sex and race:** The analysis reveals unadjusted differences in income proportions across sex and race categories. These are descriptive associations and do not account for other potentially related variables.

8. **Native country:** The observed association is weak, and sparse categories violate expected-frequency assumptions, limiting the reliability of the test.

9. **Sampling weight:** `fnlwgt` shows no meaningful distributional difference between the two income groups (p = 0.1897, r = -0.0080).

---

## 7. Limitations and Considerations

Several limitations should be considered when interpreting the findings:

- **Bivariate analysis:** Each feature was investigated independently against income. The analysis does not account for interactions or confounding between features.
- **No causal inference:** The findings describe associations and cannot establish cause-and-effect relationships.
- **Missing-value representation:** Actual `NaN` values and `?` placeholders were treated as separate categories during the analysis.
- **Sparse categorical groups:** Some categories contain very few observations, particularly within `native-country`, which affects the reliability of proportions and statistical tests.
- **Sampling weights:** The `fnlwgt` feature was not applied as a sampling weight when calculating proportions or performing statistical tests. Therefore, the findings describe the analyzed sample rather than providing weighted estimates of the 1994 U.S. population.
- **Dataset age:** The data originate from 1994, and the fixed $50,000 income threshold reflects the dataset's historical definition rather than a current income standard.
- **Capital-related features:** Both variables are highly zero-inflated, and the spike at 99,999 in `capital-gain` was observed but not investigated further. Positive-only comparisons also apply to a restricted subset.
- **Potential record dependence:** The independence assumption underlying the statistical tests was not separately investigated.
- **Multiple comparisons:** No multiple-comparison adjustment was performed. The analysis therefore does not formally account for the number of statistical tests conducted.
- **Feature redundancy:** `education` and `education-num` encode the same educational information, while `relationship` and `marital-status` contain overlapping information.
- **Sensitive attributes:** Differences involving `race`, `sex`, and `native-country` are reported descriptively and should not be interpreted as intrinsic or causal effects.

---

## 8. Conclusion

This project provides a detailed exploratory and statistical examination of the Adult Census Income dataset, covering its structure, data quality, feature distributions, and relationships with income.

The analysis identifies several meaningful associations between income and demographic, educational, employment, and financial characteristics. Household and marital features show the strongest categorical associations, followed by education and occupation. Age and working hours also demonstrate noticeable distributional differences between the two income groups.

Capital gain stands out as a particularly strong indicator among observations with positive values, while `fnlwgt` shows no meaningful difference between income categories.

An important outcome of the project is the distinction between statistical significance and practical importance. Given the relatively large sample size, effect sizes were essential for understanding the magnitude of the observed relationships rather than relying on p-values alone.

The exploration also highlighted several data-quality and analytical considerations, including inconsistent target labels, multiple representations of missing information, class imbalance, zero-inflated financial features, sparse categories, and overlapping features.

Overall, the findings provide a foundation for the subsequent data-cleaning and preprocessing stage. They also establish important considerations that should be addressed before developing and evaluating any future machine learning models.

**No predictive modeling was performed as part of this project.**
