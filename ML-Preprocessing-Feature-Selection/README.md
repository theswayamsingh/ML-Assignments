# Data Preprocessing and Feature Selection Pipeline

## Problem Statement
Predicting customer purchase decisions using demographic and financial parameters. This represents a binary classification task to optimize marketing targeting efficiency.

## Group Members & Contributions
| Student | Roll Number | Contribution |
|---|---|---|
| Swayam Singh | CSJMA23001390061 | Exploratory Analysis, Missing Values & Stats from Scratch |
| Pranjal Sharma | CSJMA23001390063 | Outlier Detection, Categorical Encoders & Scaling |
| Utkarsh Singh | CSJMA23001390065 | Feature Selection (Variance, Correlation, Chi2, ANOVA, MI) |

## Dataset Description & Source
- **Observations:** 305 records
- **Features:** Age (Numerical), Income (Numerical), CreditScore (Numerical), ConstantFeature (Numerical), Gender (Categorical), City (Categorical), Purchased (Target).
- **Source:** Benchmark synthetic consumer records.

## From-Scratch Implementations
1. **Descriptive Statistics:** Arithmetic Mean, Median, Mode, Variance, Sample Standard Deviation, Range.
2. **Missing Value Imputation:** Train-fit Median Imputer.
3. **Cleaning:** Hash-based duplicate removal and string normalizer.
4. **Encoding:** Ordinal Label Encoding and Binary One-Hot Encoding.
5. **Outlier Treatment:** IQR Quantile interpolation and capping.
6. **Scaling:** Min-Max Normalizer and Z-score Standardization.
7. **Feature Selection:**
   - Variance Threshold ($s^2 > \tau$)
   - Pearson Correlation Matrix ($r$)
   - Chi-Square Test ($\chi^2 = \sum \frac{(O-E)^2}{E}$)
   - ANOVA F-Test ($F = \frac{MSB}{MSW}$)
   - Mutual Information ($MI(X; Y) = \sum P(x, y) \log_2 \frac{P(x,y)}{P(x)P(y)}$)

## Data Leakage Prevention Strategy
All preprocessing estimators (imputation statistics, scaling means/standard deviations, outlier bounds, and one-hot categories) are fitted strictly on `df_train` and then applied to `df_test`.

## Before vs After Preprocessing Summary
| Parameter | Before Preprocessing | After Preprocessing (Encoded) | After Feature Selection (Final) |
|---|---|---|---|
| **Total Rows** | 305 | 300 (Duplicates Removed) | 300 |
| **Input Features** | 6 | 9 (OHE expanded) | 8 (`ConstantFeature` dropped) |
| **Target Variable** | 1 (`Purchased`) | 1 (`Purchased`) | 1 (`Purchased`) |
| **Total CSV Columns** | 7 | 10 | 9 |
| **Missing Values** | 35 | 0 (Imputed via Training Median) | 0 |
| **Low-Variance Features** | 1 (`ConstantFeature`) | 1 | 0 (Removed `ConstantFeature`) |

## Setup and Execution
1. Clone the repository:
   ```bash
   git clone https://github.com/theswayamsingh/ML-Assignments.git
   ```
2. Install dependencies:
   ```bash
   cd ML-Preprocessing-Feature-Selection

   pip install numpy pandas matplotlib seaborn scipy scikit-learn
   ```

3. Run the complete notebook:

    Open notebooks/main_analysis.ipynb in Jupyter Notebook or Google Colab and run all cells.

## Google Colab Link
https://colab.research.google.com/github/theswayamsingh/ML-Assignments/blob/main/ML-Preprocessing-Feature-Selection/notebooks/main_analysis.ipynb
   
