import pandas as pd, numpy as np, json, joblib, warnings
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import *
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
warnings.filterwarnings("ignore"); sns.set_theme(style="whitegrid")
R = {}
df = pd.read_csv("data/telco.csv")
R["shape_raw"] = list(df.shape)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan))
R["missing"] = df.isna().sum()[lambda s: s > 0].to_dict()
R["duplicates"] = int(df.drop(columns="customerID").duplicated().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)   # tenure=0 -> belum ditagih
df["Churn"] = (df["Churn"] == "Yes").astype(int)
R["churn_rate"] = float(df.Churn.mean()); R["churn_counts"] = df.Churn.value_counts().to_dict()
# outlier IQR
out = {}
for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    out[c] = int(((df[c] < q1 - 1.5*i) | (df[c] > q3 + 1.5*i)).sum())
R["outliers_iqr"] = out
# ---- EDA figures
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
df.Churn.map({0: "Tidak Churn", 1: "Churn"}).value_counts().plot.bar(ax=ax[0], color=["#4C78A8", "#E45756"], rot=0)
ax[0].set_title("Distribusi Kelas Target")
sns.histplot(data=df, x="tenure", hue=df.Churn.map({0: "Tidak", 1: "Churn"}), bins=30, ax=ax[1], multiple="stack")
ax[1].set_title("Distribusi Tenure"); plt.tight_layout(); plt.savefig("figures/eda1.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, ["Contract", "InternetService", "PaymentMethod"]):
    df.groupby(c).Churn.mean().sort_values().plot.barh(ax=a, color="#E45756"); a.set_title(f"Churn rate per {c}"); a.set_xlabel("")
plt.tight_layout(); plt.savefig("figures/eda2.png", dpi=150); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax[0]); sns.boxplot(data=df, x="Churn", y="TotalCharges", ax=ax[1])
plt.tight_layout(); plt.savefig("figures/eda3.png", dpi=150); plt.close()
# ---- Feature engineering
svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = sum((df[c] == "Yes").astype(int) for c in svc)
df["AvgCharge"] = df.TotalCharges / df.tenure.clip(lower=1)
df["ChargeDiff"] = df.MonthlyCharges - df.AvgCharge
df["TenureGroup"] = pd.cut(df.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"]).astype(str)
df["AutoPay"] = df.PaymentMethod.str.contains("automatic").astype(int)
for c in ["MonthlyCharges", "TotalCharges"]:  # winsorizing ringan
    lo, hi = df[c].quantile([.01, .99]); df[c] = df[c].clip(lo, hi)
X = df.drop(columns=["customerID", "Churn"]); y = df.Churn
num = ["tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgCharge", "ChargeDiff"]
cat = [c for c in X.columns if c not in num]
num_corr = pd.concat([X[num], y], axis=1).corr()
plt.figure(figsize=(6, 5)); sns.heatmap(num_corr, annot=True, fmt=".2f", cmap="coolwarm"); plt.title("Korelasi Fitur Numerik"); plt.tight_layout(); plt.savefig("figures/eda4.png", dpi=150); plt.close()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
pre = ColumnTransformer([("n", StandardScaler(), num), ("c", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
def pipe(m): return Pipeline([("pre", pre), ("smote", SMOTE(random_state=42)), ("m", m)])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cfg = {
 "Logistic Regression": (GridSearchCV, pipe(LogisticRegression(max_iter=2000)), {"m__C": [.01, .1, 1, 10], "m__penalty": ["l2"]}),
 "Random Forest": (RandomizedSearchCV, pipe(RandomForestClassifier(random_state=42, n_jobs=1)),
    {"m__n_estimators": [100, 200], "m__max_depth": [5, 8, 12, None], "m__min_samples_leaf": [1, 3, 5, 10], "m__max_features": ["sqrt", .3]}),
 "Gradient Boosting": (RandomizedSearchCV, pipe(GradientBoostingClassifier(random_state=42)),
    {"m__n_estimators": [100, 150], "m__learning_rate": [.05, .1], "m__max_depth": [2, 3], "m__subsample": [.7, .85, 1.0]}),
 "SVM (RBF)": (RandomizedSearchCV, pipe(SVC(probability=True, random_state=42, cache_size=500)), {"m__C": [.5, 1, 5], "m__gamma": ["scale", .01]}),
}
res, best, fig_roc = {}, {}, plt.figure(figsize=(6, 5))
for name, (S, p, g) in cfg.items():
    kw = dict(n_iter=4, random_state=42) if S is RandomizedSearchCV else {}
    s = S(p, g, scoring="f1", cv=cv, n_jobs=1, **kw).fit(Xtr, ytr)
    pr = s.predict_proba(Xte)[:, 1]; pd_ = (pr >= .5).astype(int)
    res[name] = dict(cv_f1=float(s.best_score_), accuracy=accuracy_score(yte, pd_), precision=precision_score(yte, pd_),
        recall=recall_score(yte, pd_), f1=f1_score(yte, pd_), roc_auc=roc_auc_score(yte, pr), pr_auc=average_precision_score(yte, pr),
        best_params={k.replace("m__", ""): (v if not isinstance(v, (np.generic,)) else v.item()) for k, v in s.best_params_.items()})
    best[name] = s.best_estimator_; fpr, tpr, _ = roc_curve(yte, pr); plt.plot(fpr, tpr, label=f"{name} (AUC={res[name]['roc_auc']:.3f})")
    print(name, {k: round(v, 3) for k, v in res[name].items() if k != "best_params"}, res[name]["best_params"], flush=True)
plt.plot([0, 1], [0, 1], "k--"); plt.legend(loc="lower right", fontsize=8); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Kurva ROC (Data Uji)")
plt.tight_layout(); plt.savefig("figures/roc.png", dpi=150); plt.close()
# pilih model terbaik berdasar F1 uji (kelas minoritas) -> tie-break ROC-AUC
win = max(res, key=lambda k: (res[k]["f1"], res[k]["roc_auc"])); R["winner"] = win
# threshold tuning pada model terbaik (dipilih di CV out-of-fold, bukan di data uji)
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(best[win], Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
ths = np.arange(.2, .8, .02); f1s = [f1_score(ytr, oof >= t) for t in ths]; th = float(ths[int(np.argmax(f1s))]); R["threshold"] = th
pr = best[win].predict_proba(Xte)[:, 1]; pd_ = (pr >= th).astype(int)
R["final_test"] = dict(accuracy=accuracy_score(yte, pd_), precision=precision_score(yte, pd_), recall=recall_score(yte, pd_), f1=f1_score(yte, pd_), roc_auc=roc_auc_score(yte, pr))
cm = confusion_matrix(yte, pd_); R["cm"] = cm.tolist()
ConfusionMatrixDisplay(cm, display_labels=["Tidak Churn", "Churn"]).plot(cmap="Blues", values_format="d"); plt.title(f"Confusion Matrix - {win} (th={th:.2f})")
plt.tight_layout(); plt.savefig("figures/cm.png", dpi=150); plt.close()
# feature importance
mdl = best[win].named_steps["m"]; names = best[win].named_steps["pre"].get_feature_names_out()
imp = getattr(mdl, "feature_importances_", None)
if imp is None: imp = np.abs(getattr(mdl, "coef_", np.zeros((1, len(names))))[0])
top = pd.Series(imp, index=[n.split("__")[1] for n in names]).sort_values(ascending=False).head(12)
top.iloc[::-1].plot.barh(color="#4C78A8", figsize=(7, 5)); plt.title(f"Top 12 Fitur - {win}"); plt.tight_layout(); plt.savefig("figures/imp.png", dpi=150); plt.close()
R["top_features"] = top.round(4).to_dict(); R["models"] = res
R["n_train"], R["n_test"] = len(Xtr), len(Xte)
joblib.dump(dict(model=best[win], threshold=th, name=win), "models/churn_model.joblib")
json.dump(R, open("results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ["winner", "threshold", "final_test", "cm", "top_features", "outliers_iqr", "missing", "duplicates", "churn_rate"]}, indent=1, default=float))
