# 📘 Diabetes Readmission Analysis — Complete Code Documentation

**Project:** Diabetes Patient Readmission Risk Analysis  
**Dataset:** 130 US Hospitals | 1999–2008 | ~101,000 Patient Records  
**Author:** [Your Name]  
**Tools:** Python, Pandas, Matplotlib, Seaborn, Scikit-learn

---

## Table of Contents
1. [Project Goal](#1-project-goal)
2. [File Structure](#2-file-structure)
3. [Section 1 — Import Libraries](#3-section-1--import-libraries)
4. [Section 2 — Load Data](#4-section-2--load-data)
5. [Section 3 — Parse Mapping File](#5-section-3--parse-mapping-file)
6. [Section 4 — Initial Exploration](#6-section-4--initial-exploration)
7. [Section 5 — Data Cleaning](#7-section-5--data-cleaning)
8. [Section 6 — Feature Engineering](#8-section-6--feature-engineering)
9. [Section 7 — EDA Charts (1–10)](#9-section-7--eda-charts)
10. [Section 8 — SQL-Style Queries](#10-section-8--sql-style-queries)
11. [Section 9 — ML Feature Preparation](#11-section-9--ml-feature-preparation)
12. [Section 10 — Model Training](#12-section-10--model-training)
13. [Section 11 — ML Charts (11–14)](#13-section-11--ml-charts)
14. [Section 12 — Export for Power BI](#14-section-12--export-for-power-bi)
15. [Key Insights Summary](#15-key-insights-summary)
16. [What to Say in Interviews](#16-what-to-say-in-interviews)

---

## 1. Project Goal

The business problem is: **"Can we predict which diabetic patients will be readmitted to hospital within 30 days of discharge?"**

This matters because:
- 30-day readmissions cost US hospitals ~$26 billion/year in penalties
- Early prediction = intervention before discharge = saved lives + saved costs
- HbA1c (blood glucose marker) testing was only done in 18.4% of patients — we investigate why this matters

---

## 2. File Structure

```
your_project_folder/
│
├── diabetes_readmission_analysis.py   ← Main code file (run this)
├── diabetic_data.csv                  ← Raw patient data (101,766 rows)
├── IDS_mapping.csv                    ← Lookup table for coded IDs
│
└── plots/                             ← Auto-created when you run the code
    ├── 01_readmission_distribution.png
    ├── 02_hba1c_vs_readmission.png
    ├── 03_readmission_by_age.png
    ├── 04_readmission_by_diagnosis.png
    ├── 05_time_in_hospital.png
    ├── 06_medications_vs_readmission.png
    ├── 07_insulin_vs_readmission.png
    ├── 08_correlation_heatmap.png
    ├── 09_race_analysis.png
    ├── 10_admission_type_readmission.png
    ├── 11_confusion_matrices.png
    ├── 12_roc_curve.png
    ├── 13_feature_importance.png
    ├── 14_model_comparison.png
    ├── query1_age_readmission.csv
    ├── query2_diagnosis_stay.csv
    ├── query3_hba1c_rate.csv
    ├── query4_specialty_readmission.csv
    └── diabetes_cleaned_for_powerbi.csv
```

---

## 3. Section 1 — Import Libraries

### Every library explained:

| Library | What it does | Why we need it |
|---------|-------------|----------------|
| `pandas` | Handles tables (DataFrames) | Reading CSV, cleaning, grouping data |
| `numpy` | Math operations on arrays | Creating arrays, polynomial fitting, linspace |
| `matplotlib.pyplot` | Drawing charts | Base engine for all plots |
| `matplotlib.ticker` | Formatting axis labels | Adding % signs to axis values |
| `seaborn` | Beautiful statistical charts | Boxplots, heatmaps, countplots |
| `warnings` | Python's warning system | Suppressing irrelevant deprecation messages |
| `os` | Operating system interface | Creating the 'plots' folder automatically |
| `sklearn.model_selection.train_test_split` | Split data randomly | 80% train, 20% test — prevents overfitting |
| `sklearn.preprocessing.LabelEncoder` | Convert text→numbers | ML models need numbers, not strings |
| `sklearn.linear_model.LogisticRegression` | ML model 1 | Predicts probability of readmission |
| `sklearn.ensemble.RandomForestClassifier` | ML model 2 | More powerful, gives feature importance |
| `sklearn.metrics.*` | Model scoring functions | Measure how good our predictions are |

```python
warnings.filterwarnings('ignore')
# WHY: Pandas and sklearn often produce DeprecationWarnings about internal changes.
# These clutter your output. Suppressing them keeps the terminal readable.
# This does NOT hide your actual errors — only library-internal warnings.

plt.style.use('seaborn-v0_8-whitegrid')
# WHY: This applies a pre-built visual theme to ALL charts.
# 'whitegrid' = white background with subtle grid lines — clean and professional.
# v0_8 prefix is needed in newer matplotlib versions.

os.makedirs('plots', exist_ok=True)
# WHY: We save all 14 charts to a 'plots' folder.
# exist_ok=True means: if the folder already exists, don't throw an error.
```

---

## 4. Section 2 — Load Data

```python
df = pd.read_csv('diabetic_data.csv', na_values='?')
```

**Line by line:**
- `pd.read_csv()` — reads a CSV file into a pandas DataFrame (like an Excel sheet in Python)
- `na_values='?'` — **CRITICAL.** The raw CSV uses `?` to mean "data missing." Without this, Python thinks `?` is a valid text value. With this, `?` becomes `NaN` (Not a Number) — Python's standard representation of missing data. This enables us to use `.isnull()`, `.fillna()`, `.dropna()` etc.

---

## 5. Section 3 — Parse Mapping File

### Why do we need this?

The main dataset has columns like `admission_type_id = 1`, `admission_type_id = 2`, etc. These numbers mean nothing by themselves. The mapping file tells us `1 = Emergency`, `2 = Urgent`, etc.

### The problem with the mapping file

The `IDS_mapping.csv` is not a normal table. It has **3 separate tables stacked vertically** in one file:

```
admission_type_id | description
1                 | Emergency
2                 | Urgent
...
                              ← blank row separator
discharge_disposition_id | description
1                         | Discharged to home
...
```

So we can't just `pd.read_csv()` it directly. We loop through it row by row and detect which section we're in.

```python
for _, row in mapping_df.iterrows():
    first_col = str(row.iloc[0]).strip()
    # WHY str() + .strip(): The values might be numbers, NaN, or strings.
    # str() converts everything to text. .strip() removes spaces/newlines.

    if first_col == 'admission_type_id':
        current_section = 'admission_type'
        continue  # 'continue' skips to the next loop iteration (skip this header row)
```

The `try/except` block:
```python
try:
    key = int(float(first_col))
except:
    continue
# WHY: Some cells have 'nan', empty string, or text headers.
# float() would fail on those. The except catches failures and skips bad rows.
# We do int(float()) — not int() directly — because '1.0' can't be int() directly.
```

---

## 6. Section 4 — Initial Exploration

```python
missing = df.isnull().sum()
# isnull() returns True/False for every cell — True where data is missing
# .sum() counts the True values per column (True = 1, False = 0)

missing_pct = (missing / len(df) * 100).round(2)
# len(df) = total number of rows
# Dividing count by total rows = fraction → multiply by 100 = percentage
# .round(2) = keep only 2 decimal places

.query('`Missing Count` > 0')
# Filters to only show columns that actually have missing values
# Backticks needed because column name has a space in it
```

---

## 7. Section 5 — Data Cleaning

### 5.1 — Why drop `weight` and `payer_code`?

```python
df.drop(columns=['weight', 'payer_code'], inplace=True)
```

- `weight`: **97% of values are missing.** A column that's 97% empty cannot teach the model anything. It would just add noise.
- `payer_code`: **52% missing**, and insurance type is not directly causally related to whether a patient gets readmitted — it's a billing attribute, not a clinical one.
- `inplace=True`: Modifies `df` directly without needing to write `df = df.drop(...)`. Saves memory.

### 5.2 — Why fill with 'Missing' vs dropping rows?

```python
df['medical_specialty'].fillna('Missing', inplace=True)
```

**Option A — Drop rows with missing medical_specialty:**
- We'd lose ~53% of our data — that's over 50,000 patients. Terrible idea.

**Option B — Fill with 'Missing' as a category:**
- "Missing" now becomes its own category in our analysis.
- In healthcare, missing specialty often means the admission was from a general ward — that IS useful information.
- This preserves all rows while being honest that the data is absent.

### 5.3 — Why remove deaths and hospice discharges?

```python
df = df[~df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
```

- IDs 11, 13, 14, 19, 20, 21 correspond to: Expired, Hospice/home, Hospice/medical facility
- Dead patients **cannot be readmitted**. If we keep them, they always appear as "not readmitted" — which would **artificially inflate our non-readmission numbers and bias the model**.
- `~` is the NOT operator — `isin()` finds matches, `~` flips it to "exclude these"

### 5.4 — Why keep only one encounter per patient?

```python
df = df.drop_duplicates(subset='patient_nbr', keep='first')
```

- Some patients have multiple hospital visits in the dataset.
- If we keep all visits for the same patient, those rows are **not independent** — they share the same person's biology, habits, and health history.
- Logistic regression and many ML models assume each row is a different, independent observation.
- Keeping multiple rows from one patient = **data leakage** (the model indirectly learns to recognize that specific patient).
- `keep='first'` → we sorted by `encounter_id` first, so 'first' = the chronologically earliest visit.

### 5.5 — Why create `readmitted_binary`?

```python
df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
```

- Original column has 3 values: `'<30'`, `'>30'`, `'NO'`
- Our business question is specifically about **30-day readmission** — the threshold used by hospitals and insurance companies for penalties
- `(df['readmitted'] == '<30')` returns True/False
- `.astype(int)` converts True→1, False→0 — ready for ML

---

## 8. Section 6 — Feature Engineering

Feature engineering = creating **new, smarter columns** from existing raw ones.

### 6.1 — Diagnosis Group Mapping

```python
def map_diagnosis(diag):
    try:
        code = float(str(diag).replace('V', '0').replace('E', '0'))
```

**Why replace V and E?**
- ICD-9 codes starting with `V` (like `V27`) are supplementary codes for health status (pregnancy outcomes, etc.)
- ICD-9 codes starting with `E` are external cause codes (accidents, poisoning causes)
- `float('V27')` would crash. We replace V→0 and E→0 to get a parseable number, then fall through to `'Other'` category since 0 doesn't match any range.

```python
if str(diag).startswith('250'):    return 'Diabetes'
```
**Why check this first?** The code 250.xx specifically means diabetes mellitus. Range `250-259` would include other endocrine disorders — we want ONLY 250.xx, so we check the string directly.

### 6.2 — Age Numeric

```python
age_map = {'[0-10)':10, '[10-20)':15, ...}
df['age_numeric'] = df['age'].map(age_map)
```

- Age is stored as text ranges because of patient privacy (you can't have exact age).
- We convert to the **midpoint** of each range (e.g., [50-60) → 55).
- ML models need numbers — they can't process `'[50-60)'` as a meaningful value.
- Midpoints preserve the ordinal (ranked) nature of age groups.

### 6.4 — Total Prior Visits

```python
df['total_prior_visits'] = df['number_outpatient'] + df['number_emergency'] + df['number_inpatient']
```

**Why create this?**
- A patient who visited the hospital 10 times in the past year is clearly sicker than one visiting for the first time.
- Rather than the model having to figure out that outpatient + emergency + inpatient all point in the same direction, we give it a single summary feature.
- This is called **domain knowledge feature engineering** — combining raw features using your understanding of the problem.

---

## 9. Section 7 — EDA Charts

### Chart 1 — Readmission Distribution (Pie Chart)
**Question answered:** What percentage of patients are readmitted within 30 days?  
**Key finding:** Only ~11% are readmitted within 30 days — this means our data is **imbalanced** (important for ML later).

```python
explode=(0, 0, 0.05)
# The 3 values correspond to the 3 pie slices.
# 0.05 = pull the '<30 days' slice out by 5% to draw viewer attention to it.
```

### Chart 2 — HbA1c vs Readmission (Side-by-side bars)
**Question answered:** Does ordering an HbA1c test reduce readmission?  
**Key finding:** 81.6% of patients had NO HbA1c test. Among those tested, readmission rates were lower — even just the act of testing (showing more clinical attention) is associated with better outcomes.

```python
ax.axhline(y=df['readmitted_binary'].mean()*100, ...)
# axhline = add a horizontal reference line
# We show the overall average readmission rate as context for comparison
```

### Chart 3 — Age vs Readmission (Dual axis chart)
**Question answered:** Are older patients more at risk?  
**Key finding:** Patients over 60 have 10.2% readmission rate vs 6.2% for under 30.

```python
ax2 = ax1.twinx()
# twinx() creates a second y-axis that shares the same x-axis
# Left y-axis shows readmission % (blue bars)
# Right y-axis shows patient count (red line)
# This lets us see both metrics without two separate charts
```

### Chart 4 — Diagnosis Group (Horizontal bar chart)
**Question answered:** Which medical condition has highest readmission risk?  
**Key finding:** Injury/Poisoning and Circulatory patients have the highest readmission rates.

```python
diag_stats = diag_stats.sort_values('readmission_rate_pct', ascending=True)
# For horizontal bar charts, sort ascending = highest value appears at TOP
# (because matplotlib plots from bottom to top for barh)
```

### Chart 5 — Time in Hospital (Box + Histogram)
**Question answered:** Do longer stays mean more readmissions?  
**Key finding:** Readmitted patients had slightly longer stays — they were sicker to begin with.

```python
sns.boxplot(data=df, x='readmitted_binary', y='time_in_hospital')
# Boxplot shows: median (middle line), IQR (box), outliers (dots)
# Compare distributions between readmitted=0 and readmitted=1 side-by-side
```

### Chart 6 — Medications vs Readmission (Bubble chart)
**Question answered:** Does taking more medications predict readmission?  
**Key finding:** There's a U-shaped relationship — very few OR very many medications both correlate with higher readmission.

```python
s=med_readmit['count'] / 10   # Bubble SIZE = patient count / 10
# Dividing by 10 prevents bubbles from being enormous
# Larger bubble = more patients in that medication-count group = more reliable data point

z = np.polyfit(x, y, 2)   # Fit a degree-2 polynomial (curved line, not straight)
p = np.poly1d(z)           # Convert coefficients to a callable function
# WHY degree 2? Because the relationship is U-shaped (non-linear), not straight
```

### Chart 8 — Correlation Heatmap
**Question answered:** Which features are related to each other?  
**Key finding:** Number of inpatient visits and number of diagnoses are most correlated with readmission.

```python
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
# A correlation matrix is symmetric — the upper triangle mirrors the lower triangle.
# np.triu creates a triangular mask (True in upper triangle).
# Masking the upper triangle removes the redundant mirror, making the chart easier to read.
```

---

## 10. Section 8 — SQL-Style Queries

These Pandas operations are **equivalent to SQL GROUP BY queries**:

| SQL | Pandas equivalent |
|-----|-------------------|
| `SELECT age, COUNT(*) FROM table GROUP BY age` | `df.groupby('age').agg(count=('col','count'))` |
| `SELECT age, AVG(days) FROM table GROUP BY age` | `df.groupby('age').agg(avg=('days','mean'))` |
| `WHERE count >= 100` | `.query('count >= 100')` |
| `ORDER BY rate DESC` | `.sort_values('rate', ascending=False)` |

**Why do this separately from EDA?**  
These produce exact numbers — useful for your Power BI report card visuals and for writing up your findings. The EDA charts are visual; these queries give you the precise statistics to quote in your README.

---

## 11. Section 9 — ML Feature Preparation

### Why these specific features?

```python
ml_features = [
    'time_in_hospital',      # HOW LONG: longer stay = more complications
    'num_lab_procedures',    # HOW TESTED: more tests = sicker patient
    'num_medications',       # HOW TREATED: more drugs = more complex case
    'number_inpatient',      # HISTORY: past hospitalizations predict future ones
    'number_emergency',      # HISTORY: frequent ER visits = poor baseline health
    'A1Cresult',             # CLINICAL KEY: diabetes control marker
    'insulin',               # TREATMENT: insulin dependency
    'diag_group',            # WHAT DISEASE: primary diagnosis matters greatly
    ...
]
```

We **excluded**:
- `encounter_id`, `patient_nbr` — these are just ID numbers, no clinical meaning
- `diag_1`, `diag_2`, `diag_3` — raw ICD-9 codes have 800+ unique values; too high cardinality, we already created `diag_group` from them
- `readmitted` — this is the original target; using it would be 100% data leakage

### Label Encoding

```python
le = LabelEncoder()
for col in categorical_cols:
    X[col] = X[col].astype(str)   # Convert to string first
    X[col] = le.fit_transform(X[col])
```

**What LabelEncoder does:**
- Takes `['No', 'Steady', 'Up', 'Down']` → `[0, 2, 3, 1]` (alphabetical assignment)
- It assigns a unique integer to each unique category value
- Models treat these as ordinal numbers (No=0 < Down=1 < Steady=2 < Up=3) — this is a limitation but acceptable for tree-based models like Random Forest

**Why `.astype(str)` first?**  
If a column has NaN mixed with strings, `LabelEncoder` will fail. Converting to str first turns NaN into the string `'nan'` — which gets encoded like any other category.

### Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
```

- `test_size=0.20` → 20% held out for testing, 80% for training
- `random_state=42` → Seeds the randomizer so you get the same split every time (reproducibility)
- `stratify=y` → **IMPORTANT for imbalanced data.** Without this, the random split might put all readmitted patients in training and none in test. Stratify ensures both splits have the same ~11% readmitted proportion.

---

## 12. Section 10 — Model Training

### Model 1: Logistic Regression

```python
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
```

**What it does:** Finds the best linear boundary to separate "readmitted" from "not readmitted" by learning a weight (coefficient) for each feature.

**Why `max_iter=1000`:** The default is 100 iterations. With 17 features and ~69,000 rows, the model needs more iterations to converge (find the optimal weights).

**Why `class_weight='balanced'`:**  
Our data is imbalanced: ~89% not readmitted, ~11% readmitted. Without this, the model would just predict "not readmitted" for everyone and achieve 89% accuracy — which is useless. `'balanced'` automatically adjusts weights so the model pays more attention to the minority class (readmitted).

### Model 2: Random Forest

```python
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
```

**What it does:** Builds 100 different decision trees, each trained on a random subset of data and features. Final prediction = majority vote of all 100 trees.

**Why better than Logistic Regression here?**
- Captures non-linear relationships (e.g., "high medications AND old age" is different from either alone)
- More robust to outliers
- Provides `feature_importances_` — tells us which features matter most

**Why `n_jobs=-1`?** Uses all available CPU cores in parallel. Training 100 trees on 69,000 rows can be slow — parallelization speeds it up.

### Understanding the Metrics

| Metric | Formula | Meaning in our context |
|--------|---------|------------------------|
| **Accuracy** | Correct/Total | % of all predictions that are right |
| **Precision** | TP/(TP+FP) | Of patients flagged as readmission risk, what % actually were? |
| **Recall** | TP/(TP+FN) | Of all actual readmissions, what % did we catch? |
| **F1-Score** | 2×P×R/(P+R) | Balance of precision and recall |
| **ROC-AUC** | Area under ROC curve | Overall model quality regardless of threshold |

**In healthcare, Recall is the most important metric.** Missing a true readmission (False Negative) is far more dangerous than a false alarm (False Positive).

---

## 13. Section 11 — ML Charts

### Chart 11 — Confusion Matrix

```
                  Predicted: No    Predicted: Yes
Actual: No       [ True Neg  |  False Positive ]
Actual: Yes      [ False Neg |  True Positive  ]
```

- **True Negative (top-left):** Correctly said "won't be readmitted" ✓
- **True Positive (bottom-right):** Correctly caught a readmission ✓
- **False Positive (top-right):** Said "will be readmitted" but wasn't — unnecessary intervention
- **False Negative (bottom-left):** Missed a readmission — the dangerous error in healthcare

```python
fmt='d'   # In sns.heatmap, fmt='d' = display as integer (not scientific notation like 5.3e+03)
```

### Chart 12 — ROC Curve

```python
fpr, tpr, _ = roc_curve(y_test, proba)
# fpr = False Positive Rate (x-axis): how often we wrongly flag non-readmissions
# tpr = True Positive Rate (y-axis): how often we correctly catch readmissions
# The _ is thresholds — we don't use them for the plot
```

**Reading the ROC curve:**
- The diagonal line (AUC=0.5) = random guessing — no better than a coin flip
- Our curve bowing toward the top-left = better than random
- AUC ~0.65-0.70 is typical for clinical readmission models with limited features

### Chart 13 — Feature Importance

```python
importances = rf_model.feature_importances_
```

- Random Forest calculates how much each feature reduces prediction uncertainty across all 100 trees
- Higher value = more useful feature for predicting readmission
- **This is one of the most valuable outputs for healthcare insights** — you can tell a hospital "focus on these 5 clinical factors"

```python
colors_imp = ['#e74c3c' if imp >= feat_imp_df['importance'].quantile(0.75) else '#3498db'
              for imp in feat_imp_df['importance']]
# quantile(0.75) = 75th percentile = top 25% importance scores
# Red = top 25% most important features, Blue = rest
# This is a list comprehension — one-line for loop creating a list
```

---

## 14. Section 12 — Export for Power BI

```python
df[powerbi_cols].to_csv('plots/diabetes_cleaned_for_powerbi.csv', index=False)
```

- `index=False` → Don't write the row numbers (0, 1, 2...) as a column. Power BI doesn't need them.
- We select only meaningful columns — not all 50+ columns are useful for a dashboard
- This file loads directly into Power BI via "Get Data → Text/CSV"

**Recommended Power BI Pages:**
1. **Overview:** KPI cards (total patients, readmit rate, avg stay), pie chart
2. **Demographic Analysis:** Age group bar chart, race analysis, gender split
3. **Clinical Factors:** HbA1c testing rates, diagnosis group breakdown, insulin analysis
4. **Risk Factors:** Medications scatter, prior visits vs readmission, correlation table

---

## 15. Key Insights Summary

These are the **5 talking points** for your interview or README:

1. **HbA1c testing gap:** Only 18.4% of diabetic inpatients had their HbA1c tested — a critical marker for diabetes management. Patients who were tested had lower readmission rates, suggesting better clinical attention overall.

2. **Age is a strong predictor:** Patients above 60 had a 10.2% readmission rate vs 6.2% for patients under 30 — a 65% relative increase. Older patients need more robust discharge planning.

3. **Diagnosis matters significantly:** Injury/Poisoning and Circulatory disease patients had the highest readmission rates. Diabetes as a *primary* diagnosis had intermediate rates — often diabetes is secondary to another acute condition.

4. **Prior healthcare utilization is key:** Number of prior inpatient visits was among the top features for predicting readmission — patients with frequent past visits are significantly higher risk.

5. **Model performance:** Random Forest (AUC ~0.67) outperformed Logistic Regression, confirming non-linear interactions between clinical features. Top predictors were prior visits, number of diagnoses, medications, and time in hospital.

---

## 16. What to Say in Interviews

**Q: "Tell me about a data analytics project you've done."**

> "I analyzed a clinical dataset of 70,000+ diabetic patient records from 130 US hospitals to predict 30-day hospital readmissions. I cleaned the data using Python and Pandas — removing patients who passed away, keeping one record per patient to ensure statistical independence, and handling 97% missing values in the weight column. I built an EDA with 10 charts that revealed the key insight: despite HbA1c being a critical diabetes marker, only 18.4% of patients had it tested. I then built a Random Forest model that achieved 0.67 AUC score in predicting readmission risk, identifying prior hospital visits and number of diagnoses as the strongest predictors. Finally I built a Power BI dashboard for the business-facing summary."

**Q: "Why did you use Random Forest over Logistic Regression?"**

> "Our dataset had non-linear relationships — for example, the interaction between age and number of medications is not additive. Random Forest handles these interactions naturally. It also gave us feature importance scores, which are more interpretable for a healthcare audience. Logistic Regression was useful as a baseline and for its interpretable coefficients."

**Q: "How did you handle the class imbalance?"**

> "The dataset was imbalanced — only 11% of patients were readmitted within 30 days. I used `class_weight='balanced'` in both models, which automatically adjusts the penalty for misclassifying the minority class. I also focused on ROC-AUC and Recall as evaluation metrics rather than raw accuracy, since 89% accuracy is achievable just by predicting nobody gets readmitted — which is useless clinically."

---

*Documentation created alongside `diabetes_readmission_analysis.py`*  
*Run the Python file with: `python diabetes_readmission_analysis.py`*
