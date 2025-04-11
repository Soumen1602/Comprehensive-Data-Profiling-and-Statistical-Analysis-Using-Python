# Comprehensive-Data-Profiling-and-Statistical-Analysis-Using-Python
This Python project performs complete data analysis including cleaning, visualization, EDA, and statistical testing. It uses pandas, seaborn, matplotlib to handle missing data, detect outliers, visualize trends, and perform T-tests and normality checks for insights.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load dataset
df = pd.read_csv("C:/Users/soume/Downloads/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69.csv")
print("Dataset Loaded Successfully!\n")
print("First 5 Rows:\n", df.head())
print("\nData Types:\n", df.dtypes)

# ========================
# Objective 2: Data Cleaning & Manipulation (Unit II)
# ========================

print("\nMissing Values:\n", df.isnull().sum())

# Fill missing numeric values with mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Optional: clean column names
df.columns = [col.strip().replace(" ", "_") for col in df.columns]

# ========================
# Objective 3: Data Visualization (Unit III)
# ========================

# --- Histogram
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Histogram of Numerical Columns", fontsize=16)
plt.tight_layout()
plt.show()

# --- Pie Chart for a categorical column
# We'll choose the first column with <10 unique categories
cat_col = None
for col in df.columns:
    if df[col].dtype == 'object' and df[col].nunique() <= 10:
        cat_col = col
        break

if cat_col:
    plt.figure(figsize=(6, 6))
    df[cat_col].value_counts().plot.pie(autopct='%1.1f%%', startangle=140)
    plt.title(f"Distribution of {cat_col}")
    plt.ylabel("")
    plt.show()
else:
    print("No suitable categorical column found for pie chart.")

# --- Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# --- Pie Chart: Proportion of Non-null Records by Column (Like the Screenshot)
non_null_counts = df.notnull().sum()
plt.figure(figsize=(10, 8))
plt.pie(non_null_counts, labels=non_null_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Proportion of Records by Category")
plt.axis('equal')  # Equal aspect ratio ensures the pie is circular.
plt.tight_layout()
plt.show()



# --- Boxplots
num_cols = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols[:2]:  # just two for demonstration
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# ========================
# Objective 4: Exploratory Data Analysis (Unit IV)
# ========================

print("\nSummary Statistics:\n", df.describe())

# --- Outlier Detection using IQR
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
    print(f"{col}: {len(outliers)} outliers detected")

# ========================
# Objective 5: Statistical Analysis (Unit V)
# ========================

# --- T-test for gender difference in numerical data
if 'Gender' in df.columns and df['Gender'].nunique() == 2:
    genders = df['Gender'].unique()
    for col in num_cols:
        group1 = df[df['Gender'] == genders[0]][col].dropna()
        group2 = df[df['Gender'] == genders[1]][col].dropna()
        if not group1.empty and not group2.empty:
            stat, p = stats.ttest_ind(group1, group2)
            print(f"T-test on {col} between {genders[0]} and {genders[1]}: p-value = {p:.4f}")

# --- Shapiro-Wilk Normality Test
print("\nNormality Test Results (Shapiro-Wilk):")
for col in num_cols[:3]:
    stat, p = stats.shapiro(df[col])
    print(f"{col}: W = {stat:.4f}, p = {p:.4f} {'(Normal)' if p > 0.05 else '(Not Normal)'}")

# ========================
# Objective 6: Correlation with Key Variable (Optional)
# ========================

if 'Amount_Spent' in df.columns:
    corr_target = df.corr(numeric_only=True)['Amount_Spent'].sort_values(ascending=False)
    print("\nTop Features Correlated with Amount_Spent:\n", corr_target)
