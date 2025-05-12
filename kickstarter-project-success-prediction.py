#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1: EDA & Data Loading
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from time import strftime, localtime

def stamp(msg):
    print(f"[{strftime('%Y-%m-%d %H:%M:%S', localtime())}] {msg}")

DATA_PATH = "/Users/alan/Downloads/kickstarter_data_with_features.csv"
stamp(f"Loading data from {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
stamp("Dataset loaded")

# Filter to only completed campaigns
df = df[df["state"].isin(["successful","failed"])].copy()
stamp(f"Filtered: {df.shape[0]} rows")

# Basic info
stamp("Data head")
display(df.head())
stamp("Data info")
df.info()

stamp("Top 10 missing % columns")
missing = (df.isnull().mean()*100).sort_values(ascending=False).head(10)
display(missing.to_frame("missing_%"))

# Plot outcome distribution
stamp("Plotting outcome distribution")
sns.countplot(x="state", data=df)
plt.title("Successful vs. Failed")
plt.show()

# Plot log(goal)
stamp("Plotting log(goal + 1) histogram")
sns.histplot(np.log1p(df["goal"]), bins=30, kde=True)
plt.title("Log-scaled Campaign Goal")
plt.show()

# Plot success rate by category
stamp("Plotting success rate by category")
catcol = "main_category" if "main_category" in df.columns else "category"
rates = df.groupby(catcol)["state"].apply(lambda s: (s=="successful").mean()).sort_values()
plt.figure(figsize=(8,12))
rates.plot.barh()
plt.xlabel("Success Rate")
plt.show()


# In[2]:


# Cell 2: Preprocessing & Train/Test Split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition   import TruncatedSVD
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline


# Drop columns not available at launch or that leak future information:
# - Identifiers & metadata: Unnamed: 0, id, photo, slug, creator, urls, source_url  
# - Pledged amounts & state changes: pledged, usd_pledged, spotlight, state_changed_at, launch_to_state_change  
# - User-specific flags & permissions: friends, is_starred, is_backing, permissions  
# - Post‐launch timestamps (and their decomposed components): created_at*, launched_at*, deadline*  
# - Currency symbols (duplicate of currency code): currency_symbol, currency_trailing_code  
cols_to_drop = [
    "Unnamed: 0","id","photo","slug","disable_communication",
    "pledged","usd_pledged","spotlight","state_changed_at","launch_to_state_change",
    "creator","location","profile","urls","source_url","friends","is_starred",
    "is_backing","permissions",
    "state_changed_at_weekday","state_changed_at_month","state_changed_at_day",
    "state_changed_at_yr","state_changed_at_hr",
    "created_at","created_at_weekday","created_at_month","created_at_day",
    "created_at_yr","created_at_hr","launched_at","launched_at_weekday",
    "launched_at_month","launched_at_day","launched_at_yr","launched_at_hr",
    "deadline","deadline_month","deadline_day",
    "deadline_yr","deadline_hr","currency_symbol","currency_trailing_code"
]
df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

# Define X, y
y = (df["state"]=="successful").astype(int)
X = df.drop(columns="state")
stamp(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
print("Class counts:", np.bincount(y))

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
stamp(f"Train/test split: {X_train.shape[0]}/{X_test.shape[0]}")

# Feature groups:
# - numeric_log_cols: skewed numeric features for log-transform  
# - numeric_lin_cols: numeric features used as-is  
# - delta_cols: time‐delta strings to convert into days  
# - categorical_cols: low‐dimensional categorical variables  
# - text_cols: free‐text fields for TF-IDF + SVD
numeric_log_cols = ["goal", "static_usd_rate"]
numeric_lin_cols = ["name_len", "backers_count"]
delta_cols       = ["create_to_launch", "launch_to_deadline"]
categorical_cols = ["category", "country", "currency",
                    "deadline_weekday", "launched_at_weekday", "staff_pick"]
text_cols        = ["name", "blurb"]

# ── Filter each list to existing columns ──
numeric_log_cols = [c for c in numeric_log_cols    if c in X.columns]
numeric_lin_cols = [c for c in numeric_lin_cols    if c in X.columns]
delta_cols       = [c for c in delta_cols          if c in X.columns]
categorical_cols = [c for c in categorical_cols    if c in X.columns]
text_cols        = [c for c in text_cols           if c in X.columns]

# Converters
to_days = FunctionTransformer(
    lambda df: df.apply(pd.to_timedelta, errors="coerce")
                  .apply(lambda col: col.dt.total_seconds()/86400),
    feature_names_out="one-to-one"
)
merge_text = FunctionTransformer(
    lambda df: df.fillna("").agg(" ".join, axis=1),
    feature_names_out="one-to-one"
)

# Pipelines
num_log_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log",    FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale",  StandardScaler())
])
num_lin_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler())
])
delta_pipe = Pipeline([
    ("to_days", to_days),
    ("impute",  SimpleImputer(strategy="median")),
    ("scale",   StandardScaler())
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=20))
])
text_pipe = Pipeline([
    ("merge", merge_text),
    ("tfidf", TfidfVectorizer(max_features=100_000,
                              ngram_range=(1,3),
                              stop_words="english",
                              min_df=2,
                              sublinear_tf=True)),
    ("svd",   TruncatedSVD(n_components=200, random_state=42))
])

preprocessor = ColumnTransformer([
    ("num_log", num_log_pipe, numeric_log_cols),
    ("num_lin", num_lin_pipe, numeric_lin_cols),
    ("delta",   delta_pipe,   delta_cols),
    ("cat",     cat_pipe,     categorical_cols),
    ("text",    text_pipe,    text_cols)
])


# In[3]:


# Cell 3: Models (≤10 min each) + GridSearchCV(3-fold)
from sklearn.linear_model    import LogisticRegression
from sklearn.tree            import DecisionTreeClassifier
from sklearn.ensemble        import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm             import SVC
from sklearn.neural_network  import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import accuracy_score

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
results = {}

models = {
    "LogReg": (
        LogisticRegression(max_iter=500, random_state=42),
        {"clf__C": [0.5, 1.0]}
    ),
    "DecTree": (
        DecisionTreeClassifier(random_state=42),
        {"clf__max_depth": [5, 10]}
    ),
    "RandForest": (
        RandomForestClassifier(n_estimators=30, random_state=42),
        {"clf__max_depth": [None, 10]}
    ),
    "AdaBoost": (
        AdaBoostClassifier(n_estimators=30, random_state=42),
        {"clf__learning_rate": [0.5, 1.0]}
    ),
    "SVM-Lin": (
        SVC(kernel="linear", probability=True, random_state=42),
        {"clf__C": [0.5, 1.0]}
    ),
    "MLP": (
        MLPClassifier(hidden_layer_sizes=(128,),
                      learning_rate_init=1e-3,
                      max_iter=300,
                      random_state=42),
        {"clf__alpha": [1e-4, 1e-3]}
    )
}

stamp("Starting model training…")
for name,(est,grid) in models.items():
    pipe = Pipeline([("pre", preprocessor), ("clf", est)])
    if grid:
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy",
                          n_jobs=-1, verbose=0)
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
    else:
        best = pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, best.predict(X_test))
    results[name] = acc
    print(f"{name:8s} → Test Acc = {acc*100:5.2f}%")

best = max(results, key=results.get)
stamp(f"🏆 Best model: {best}  | Accuracy = {results[best]*100:.2f}%")


# In[11]:


# Cell 4: Quick Test‐Set Evaluation & Plots (optimized)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline         import Pipeline
from sklearn.metrics         import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, roc_auc_score,
    precision_recall_curve
)

# 0️⃣ Pre‐fit all models once, with AdaBoost using SAMME
fitted = {}
for name, (est, grid) in models.items():
    # if this is AdaBoost, override to use SAMME
    if name == "AdaBoost":
        est.set_params(algorithm="SAMME")
    pipe = Pipeline([("pre", preprocessor), ("clf", est)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

# 1️⃣ Confusion Matrices on TEST set
fig, axes = plt.subplots(2, len(fitted)//2, figsize=(12,8))
axes = axes.ravel()
for ax, (name, pipe) in zip(axes, fitted.items()):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["failed","success"]) \
        .plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(name)
plt.suptitle("Confusion Matrices (Test Set)", y=1.02)
plt.tight_layout()
plt.show()

# 2️⃣ ROC Curves
plt.figure(figsize=(8,6))
for name, pipe in fitted.items():
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0,1],[0,1],"k--", alpha=0.5)
plt.title("ROC Curves (Test Set)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# 3️⃣ Precision–Recall Curves
plt.figure(figsize=(8,6))
for name, pipe in fitted.items():
    if hasattr(pipe, "predict_proba"):
        y_prob = pipe.predict_proba(X_test)[:,1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = np.trapz(prec, rec)
        plt.plot(rec, prec, label=f"{name} (PR‐AUC={pr_auc:.3f})")
plt.title("Precision–Recall (Test Set)")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()
