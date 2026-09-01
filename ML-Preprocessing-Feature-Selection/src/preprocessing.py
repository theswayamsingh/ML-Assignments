import math
import random
import numpy as np
import pandas as pd



# Part B: Descriptive Statistics From Scratch

def calculate_mean(values):
    """Calculates arithmetic mean: sum(X) / n"""
    clean_vals = [v for v in values if pd.notnull(v)]
    if not clean_vals:
        return 0.0
    return sum(clean_vals) / len(clean_vals)


def calculate_median(values):
    """Calculates median by sorting and finding middle element(s)"""
    clean_vals = sorted([v for v in values if pd.notnull(v)])
    n = len(clean_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2 == 1:
        return float(clean_vals[mid])
    return (clean_vals[mid - 1] + clean_vals[mid]) / 2.0


def calculate_mode(values):
    """Calculates statistical mode using frequency counts"""
    clean_vals = [v for v in values if pd.notnull(v)]
    if not clean_vals:
        return None
    counts = {}
    for v in clean_vals:
        counts[v] = counts.get(v, 0) + 1
    max_freq = max(counts.values())
    modes = [k for k, v in counts.items() if v == max_freq]
    return modes[0]


def calculate_variance(values, ddof=1):
    """Calculates sample variance: sum((X - mean)^2) / (n - ddof)"""
    clean_vals = [v for v in values if pd.notnull(v)]
    n = len(clean_vals)
    if n <= ddof:
        return 0.0
    mean_val = calculate_mean(clean_vals)
    return sum((x - mean_val) ** 2 for x in clean_vals) / (n - ddof)


def calculate_std(values, ddof=1):
    """Calculates sample standard deviation: sqrt(variance)"""
    return math.sqrt(calculate_variance(values, ddof=ddof))


def calculate_min_max_range(values):
    """Calculates Minimum, Maximum, and Range"""
    clean_vals = [v for v in values if pd.notnull(v)]
    if not clean_vals:
        return None, None, None
    min_v, max_v = clean_vals[0], clean_vals[0]
    for x in clean_vals[1:]:
        if x < min_v:
            min_v = x
        if x > max_v:
            max_v = x
    return min_v, max_v, (max_v - min_v)



# Part C: Missing Value Analysis & Imputation From Scratch

def get_missing_summary(df):
    """Reports missing count and missing percentage for each column"""
    total_rows = len(df)
    summary = []
    for col in df.columns:
        null_count = sum(1 for v in df[col] if pd.isnull(v))
        pct = (null_count / total_rows) * 100.0
        summary.append({'Feature': col, 'Missing Values': null_count, 'Missing %': round(pct, 2)})
    return pd.DataFrame(summary)


class ScratchImputer:
    """Imputes missing values using statistics learned from training data"""
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.statistics_ = {}

    def fit(self, df, columns):
        for col in columns:
            vals = df[col].tolist()
            if self.strategy == 'mean':
                self.statistics_[col] = calculate_mean(vals)
            elif self.strategy == 'median':
                self.statistics_[col] = calculate_median(vals)
            elif self.strategy == 'mode':
                self.statistics_[col] = calculate_mode(vals)
        return self

    def transform(self, df):
        df_out = df.copy()
        for col, fill_val in self.statistics_.items():
            if col in df_out.columns:
                df_out[col] = [fill_val if pd.isnull(v) else v for v in df_out[col]]
        return df_out



# Part D: Duplicate and Inconsistent Data Cleaning

def identify_and_drop_duplicates(df):
    """Identifies and drops duplicate rows based on stringified row representations"""
    seen = set()
    unique_indices = []
    duplicate_count = 0
    
    for idx, row in df.iterrows():
        row_tuple = tuple(row.values)
        if row_tuple in seen:
            duplicate_count += 1
        else:
            seen.add(row_tuple)
            unique_indices.append(idx)
            
    df_cleaned = df.loc[unique_indices].reset_index(drop=True)
    return df_cleaned, duplicate_count


def clean_inconsistent_categories(df, column, mapping):
    """Standardizes string casings, extra whitespaces, and synonymous categories"""
    df_out = df.copy()
    cleaned_vals = []
    for val in df_out[column]:
        if pd.isnull(val):
            cleaned_vals.append(val)
            continue
        cleaned_str = str(val).strip().title()
        cleaned_vals.append(mapping.get(cleaned_str, cleaned_str))
    df_out[column] = cleaned_vals
    return df_out



# Part E: Categorical Encoding From Scratch

class ScratchLabelEncoder:
    """Maps categories to ordinal integers using sorted unique values"""
    def __init__(self):
        self.mapping_ = {}
        self.reverse_mapping_ = {}

    def fit(self, series):
        unique_vals = sorted(list(set([v for v in series if pd.notnull(v)])))
        self.mapping_ = {val: idx for idx, val in enumerate(unique_vals)}
        self.reverse_mapping_ = {idx: val for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, series):
        return [self.mapping_.get(v, -1) for v in series]


class ScratchOneHotEncoder:
    """Expands nominal categorical features into binary indicator vectors"""
    def __init__(self):
        self.categories_ = {}

    def fit(self, df, columns):
        for col in columns:
            unique_vals = sorted(list(set([v for v in df[col] if pd.notnull(v)])))
            self.categories_[col] = unique_vals
        return self

    def transform(self, df):
        df_out = df.copy()
        for col, cats in self.categories_.items():
            for cat in cats:
                new_col_name = f"{col}_{cat}"
                df_out[new_col_name] = [1 if v == cat else 0 for v in df_out[col]]
            df_out.drop(columns=[col], inplace=True)
        return df_out



# Part F: Outlier Detection and Capping From Scratch

def calculate_percentile(sorted_data, p):
    """Calculates linear-interpolated percentile"""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    pos = (p / 100.0) * (n - 1)
    low_idx = int(pos)
    high_idx = min(low_idx + 1, n - 1)
    weight = pos - low_idx
    return sorted_data[low_idx] + weight * (sorted_data[high_idx] - sorted_data[low_idx])


class ScratchIQROutlierDetector:
    """Calculates Q1, Q3, IQR and caps values beyond 1.5 * IQR bounds"""
    def __init__(self):
        self.bounds_ = {}

    def fit(self, df, columns):
        for col in columns:
            vals = sorted([v for v in df[col] if pd.notnull(v)])
            q1 = calculate_percentile(vals, 25)
            q3 = calculate_percentile(vals, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.bounds_[col] = {
                'q1': q1, 'q3': q3, 'iqr': iqr,
                'lower': lower_bound, 'upper': upper_bound
            }
        return self

    def cap(self, df):
        df_out = df.copy()
        for col, b in self.bounds_.items():
            df_out[col] = [
                b['lower'] if v < b['lower'] else (b['upper'] if v > b['upper'] else v)
                for v in df_out[col]
            ]
        return df_out


class ScratchZScoreDetector:
    """Identifies observations with |Z| > threshold based on training mean and std"""
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.params_ = {}

    def fit(self, df, columns):
        for col in columns:
            vals = df[col].tolist()
            mu = calculate_mean(vals)
            sigma = calculate_std(vals)
            self.params_[col] = {'mean': mu, 'std': sigma}
        return self

    def get_outliers(self, df):
        outliers_dict = {}
        for col, p in self.params_.items():
            mu, sigma = p['mean'], p['std']
            if sigma == 0:
                outliers_dict[col] = []
                continue
            col_outliers = [
                idx for idx, v in enumerate(df[col])
                if abs((v - mu) / sigma) > self.threshold
            ]
            outliers_dict[col] = col_outliers
        return outliers_dict



# Part G: Mathematical Data Transformation

def log_transform(values, offset=1.0):
    """Calculates log(X + offset) to compress right-skewed data"""
    return [math.log(x + offset) if x + offset > 0 else 0.0 for x in values]


def sqrt_transform(values):
    """Calculates sqrt(X) for moderately skewed counts"""
    return [math.sqrt(x) if x >= 0 else 0.0 for x in values]



# Part H: Feature Scaling From Scratch

class ScratchMinMaxScaler:
    """Transforms numerical features to range [0, 1] using X' = (X - Xmin) / (Xmax - Xmin)"""
    def __init__(self):
        self.bounds_ = {}

    def fit(self, df, columns):
        for col in columns:
            min_val, max_val, _ = calculate_min_max_range(df[col].tolist())
            self.bounds_[col] = {'min': min_val, 'max': max_val}
        return self

    def transform(self, df):
        df_out = df.copy()
        for col, b in self.bounds_.items():
            denominator = b['max'] - b['min']
            if denominator == 0:
                df_out[col] = 0.0
            else:
                df_out[col] = [(x - b['min']) / denominator for x in df_out[col]]
        return df_out


class ScratchStandardScaler:
    """Standardizes numerical features to zero mean and unit variance: Z = (X - mu) / sigma"""
    def __init__(self):
        self.params_ = {}

    def fit(self, df, columns):
        for col in columns:
            vals = df[col].tolist()
            mu = calculate_mean(vals)
            sigma = calculate_std(vals)
            self.params_[col] = {'mean': mu, 'std': sigma}
        return self

    def transform(self, df):
        df_out = df.copy()
        for col, p in self.params_.items():
            mu, sigma = p['mean'], p['std']
            if sigma == 0:
                df_out[col] = 0.0
            else:
                df_out[col] = [(x - mu) / sigma for x in df_out[col]]
        return df_out



# Part J: Train-Test Split From Scratch

def scratch_train_test_split(df, test_size=0.2, random_state=42):
    """Splits dataset into training and testing partitions using index shuffling"""
    if random_state is not None:
        random.seed(random_state)
    
    indices = list(range(len(df)))
    random.shuffle(indices)
    
    split_point = int(len(df) * (1.0 - test_size))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    
    train_df = df.iloc[train_indices].reset_index(drop=True)
    test_df = df.iloc[test_indices].reset_index(drop=True)
    return train_df, test_df