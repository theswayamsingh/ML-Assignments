import math
import numpy as np
import pandas as pd
from src.preprocessing import calculate_mean, calculate_variance



# Task M1: Variance Threshold From Scratch

def calculate_feature_variances(df, numerical_columns):
    """Computes variance for each numerical column to detect zero or near-zero variance features"""
    variances = {}
    for col in numerical_columns:
        variances[col] = calculate_variance(df[col].tolist())
    return variances


def apply_variance_threshold(df_train, df_test, numerical_columns, threshold=0.01):
    """Selects numerical features possessing sample variance strictly greater than threshold"""
    variances = calculate_feature_variances(df_train, numerical_columns)
    selected_features = [col for col, var in variances.items() if var > threshold]
    dropped_features = [col for col, var in variances.items() if var <= threshold]
    
    return selected_features, dropped_features, variances



# Task M2: Pearson Correlation From Scratch

def calculate_pearson_correlation(x, y):
    """
    Computes Pearson Correlation Coefficient:
    r = sum((x_i - mean_x) * (y_i - mean_y)) / sqrt(sum((x_i - mean_x)^2) * sum((y_i - mean_y)^2))
    """
    n = len(x)
    if n == 0 or len(y) != n:
        return 0.0
    
    mean_x = calculate_mean(x)
    mean_y = calculate_mean(y)
    
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    denom_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    denominator = math.sqrt(denom_x * denom_y)
    
    if denominator == 0:
        return 0.0
    return numerator / denominator


def scratch_correlation_matrix(df, columns):
    """Constructs a pairwise Pearson correlation DataFrame from scratch"""
    matrix = pd.DataFrame(index=columns, columns=columns, dtype=float)
    for col1 in columns:
        for col2 in columns:
            r = calculate_pearson_correlation(df[col1].tolist(), df[col2].tolist())
            matrix.loc[col1, col2] = round(r, 4)
    return matrix



# Task M3: Chi-Square Test for Categorical Features From Scratch

def scratch_chi2_test(feature_series, target_series):
    """
    Calculates Chi-Square statistic and degrees of freedom for discrete/categorical pairs:
    chi2 = sum((O - E)^2 / E), where E = (Row Total * Col Total) / Grand Total
    """
    categories = sorted(list(set(feature_series)))
    classes = sorted(list(set(target_series)))
    
    # 1. Contingency Table (Observed Frequencies)
    contingency = {cat: {cls: 0 for cls in classes} for cat in categories}
    for f_val, t_val in zip(feature_series, target_series):
        contingency[f_val][t_val] += 1
        
    grand_total = len(feature_series)
    row_totals = {cat: sum(contingency[cat].values()) for cat in categories}
    col_totals = {cls: sum(contingency[cat][cls] for cat in categories) for cls in classes}
    
    # 2. Compute Chi-Square statistic
    chi2_stat = 0.0
    for cat in categories:
        for cls in classes:
            o = contingency[cat][cls]
            e = (row_totals[cat] * col_totals[cls]) / grand_total
            if e > 0:
                chi2_stat += ((o - e) ** 2) / e
                
    # 3. Degrees of freedom = (r - 1) * (c - 1)
    df = (len(categories) - 1) * (len(classes) - 1)
    return chi2_stat, df, contingency



# Task M4: One-Way ANOVA F-Test From Scratch

def scratch_anova_f_test(numerical_feature, categorical_target):
    """
    Computes One-Way ANOVA F-statistic:
    F = MSB / MSW = (Between Group Variance) / (Within Group Variance)
    """
    grand_mean = calculate_mean(numerical_feature)
    groups = {}
    for x, y in zip(numerical_feature, categorical_target):
        if y not in groups:
            groups[y] = []
        groups[y].append(x)
        
    k = len(groups)          # Number of groups
    n_total = len(numerical_feature)  # Total observations
    
    if k <= 1 or n_total <= k:
        return 0.0, 0, 0
    
    # Sum of Squares Between (SSB)
    ssb = sum(len(vals) * (calculate_mean(vals) - grand_mean) ** 2 for vals in groups.values())
    df_between = k - 1
    msb = ssb / df_between if df_between > 0 else 0.0
    
    # Sum of Squares Within (SSW)
    ssw = sum(sum((x - calculate_mean(vals)) ** 2 for x in vals) for vals in groups.values())
    df_within = n_total - k
    msw = ssw / df_within if df_within > 0 else 0.0
    
    f_stat = msb / msw if msw > 0 else 0.0
    return f_stat, df_between, df_within



# Task M5: Mutual Information From Scratch (Discrete Variables)

def calculate_entropy(series):
    """Calculates Shannon Entropy in bits: H(Y) = -sum(P(y) * log2(P(y)))"""
    n = len(series)
    if n == 0:
        return 0.0
    counts = {}
    for val in series:
        counts[val] = counts.get(val, 0) + 1
    
    entropy = 0.0
    for count in counts.values():
        p = count / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def calculate_mutual_information(feature_series, target_series):
    """
    Calculates discrete Mutual Information:
    MI(X; Y) = sum_{x,y} P(x, y) * log2( P(x, y) / (P(x) * P(y)) )
    """
    n = len(feature_series)
    if n == 0 or len(target_series) != n:
        return 0.0
    
    # Joint and marginal frequencies
    joint_counts = {}
    x_counts = {}
    y_counts = {}
    
    for x, y in zip(feature_series, target_series):
        joint_counts[(x, y)] = joint_counts.get((x, y), 0) + 1
        x_counts[x] = x_counts.get(x, 0) + 1
        y_counts[y] = y_counts.get(y, 0) + 1
        
    mi = 0.0
    for (x, y), count in joint_counts.items():
        p_xy = count / n
        p_x = x_counts[x] / n
        p_y = y_counts[y] / n
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
            
    return mi