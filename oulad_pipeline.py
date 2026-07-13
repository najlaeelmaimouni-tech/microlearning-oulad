"""
OULAD -> feature space of the sequencing engine -> decision tree.
Real learner traces (Kuzilek, Hlosta & Zdrahal, 2017), CC BY 4.0.
"""
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

U = "/mnt/user-data/uploads/"
asmt = pd.read_csv(U + "assessments.csv")
sasm = pd.read_csv(U + "studentAssessment.csv")
sinf = pd.read_csv(U + "studentInfo.csv")
crs  = pd.read_csv(U + "courses.csv")

# ---------- LOM-like metadata of learning objects ----------
EDU = {"No Formal quals": 1, "Lower Than A Level": 2, "A Level or Equivalent": 3,
       "HE Qualification": 4, "Post Graduate Qualification": 5}
INTER = {"CMA": 1, "TMA": 2, "Exam": 3}          # interactivity level
asmt["date"] = pd.to_numeric(asmt["date"], errors="coerce")
asmt = asmt.dropna(subset=["date"])
asmt["interactivity"] = asmt["assessment_type"].map(INTER)
# difficulty: weight of the assessment, binned onto the 1-5 LOM scale
asmt["difficulty"] = pd.qcut(asmt["weight"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
# typical_time: proxy = days allotted since the previous assessment of the module
asmt = asmt.sort_values(["code_module", "code_presentation", "date"])
asmt["typical_time"] = (asmt.groupby(["code_module", "code_presentation"])["date"]
                        .diff().fillna(asmt["date"]).clip(1, 120))
# n_prereq: rank of the assessment within the module-presentation
asmt["n_prereq"] = asmt.groupby(["code_module", "code_presentation"]).cumcount()
# relative position of the object within the module presentation (context descriptor)
asmt = asmt.merge(crs, on=["code_module", "code_presentation"], how="left")
asmt["course_position"] = (asmt["date"] / asmt["module_presentation_length"]).round(3)

# ---------- learner profile ----------
sinf["prior_level"] = sinf["highest_education"].map(EDU)
sinf["role"] = pd.cut(sinf["studied_credits"], [0, 60, 120, 1000], labels=[0, 1, 2]).astype(float)
prof = sinf[["id_student", "code_module", "code_presentation", "prior_level", "role",
             "num_of_prev_attempts"]].dropna()

# ---------- expected pairs: every learner x every object of his module ----------
# A non-submission is a failure, not a missing value. Building the full grid avoids
# the survivorship bias that would otherwise make the target degenerate.
grid = prof.merge(asmt, on=["code_module", "code_presentation"], how="inner")
sasm["score"] = pd.to_numeric(sasm["score"], errors="coerce")
df = grid.merge(sasm[["id_assessment", "id_student", "score"]],
                on=["id_assessment", "id_student"], how="left")

# ---------- features known BEFORE the learner attempts the object ----------
df["retries"] = df["num_of_prev_attempts"].clip(0, 5)
df["prereq_ok"] = (df["prior_level"] >= df["n_prereq"] / 2).astype(int)
df["success"] = ((df["score"].notna()) & (df["score"] >= 40)).astype(int)

FEATS = ["prior_level", "role", "difficulty", "typical_time", "interactivity",
         "n_prereq", "prereq_ok", "course_position", "retries"]
data = df[FEATS + ["success", "id_assessment"]].dropna()
print(f"{len(data):,} sessions | {df.id_student.nunique():,} learners | "
      f"{df.id_assessment.nunique()} learning objects | success rate {data.success.mean():.1%}")

X, y = data[FEATS], data["success"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)

def rep(name, yhat, proba=None):
    return dict(Model=name,
                Accuracy=round(accuracy_score(yte, yhat), 3),
                Precision=round(precision_score(yte, yhat, zero_division=0), 3),
                Recall=round(recall_score(yte, yhat, zero_division=0), 3),
                F1=round(f1_score(yte, yhat, zero_division=0), 3),
                AUC=(round(roc_auc_score(yte, proba), 3) if proba is not None else "n/a"))

rng = np.random.default_rng(42)
res = [rep("Random", rng.integers(0, 2, len(yte)), rng.random(len(yte)))]
maj = int(ytr.mean() > 0.5)
res.append(rep("Fixed linear curriculum", np.full(len(yte), maj)))
pop = ytr.groupby(Xtr.index.map(lambda i: data.loc[i, "id_assessment"])).mean()
pp = np.array(Xte.index.map(lambda i: pop.get(data.loc[i, "id_assessment"], ytr.mean())))
res.append(rep("Popularity-based", (pp > 0.5).astype(int), pp))

grid = {d: cross_val_score(DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=42),
                           Xtr, ytr, cv=StratifiedKFold(5), scoring="roc_auc").mean() for d in range(2, 16)}
best = max(grid, key=grid.get)
tree = DecisionTreeClassifier(max_depth=best, min_samples_leaf=50, random_state=42).fit(Xtr, ytr)
res.append(rep(f"CART tree (proposed, depth={best})", tree.predict(Xte), tree.predict_proba(Xte)[:, 1]))
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(Xtr, ytr)
res.append(rep("Random forest", rf.predict(Xte), rf.predict_proba(Xte)[:, 1]))
lg = LogisticRegression(max_iter=3000).fit(Xtr, ytr)
res.append(rep("Logistic regression", lg.predict(Xte), lg.predict_proba(Xte)[:, 1]))

t1 = pd.DataFrame(res)
print("\n", t1.to_string(index=False))

blocks = {"Full model": FEATS,
          "without LOM metadata": [f for f in FEATS if f not in
                                   ("difficulty", "typical_time", "interactivity", "n_prereq", "prereq_ok", "course_position")],
          "without learner profile": [f for f in FEATS if f not in ("prior_level", "role", "retries")]}
abl = []
for nm, fs in blocks.items():
    t = DecisionTreeClassifier(max_depth=best, min_samples_leaf=50, random_state=42).fit(Xtr[fs], ytr)
    yh = t.predict(Xte[fs])
    abl.append(dict(Configuration=nm, Features=len(fs),
                    Accuracy=round(accuracy_score(yte, yh), 3),
                    F1=round(f1_score(yte, yh), 3),
                    AUC=round(roc_auc_score(yte, t.predict_proba(Xte[fs])[:, 1]), 3)))
t2 = pd.DataFrame(abl)
print("\n", t2.to_string(index=False))

imp = (pd.DataFrame({"Feature": FEATS, "Importance": tree.feature_importances_.round(3)})
       .sort_values("Importance", ascending=False))
print("\n", imp.head(6).to_string(index=False))
print("\nconfusion:\n", confusion_matrix(yte, tree.predict(Xte)))

t1.to_csv("real_models.csv", index=False)
t2.to_csv("real_ablation.csv", index=False)
imp.head(6).to_csv("real_importance.csv", index=False)

C = "#156082"
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for nm, m in [("CART decision tree", tree), ("Random forest", rf), ("Logistic regression", lg)]:
    fpr, tpr, _ = roc_curve(yte, m.predict_proba(Xte)[:, 1])
    ax.plot(fpr, tpr, lw=1.8, label=f"{nm} (AUC = {roc_auc_score(yte, m.predict_proba(Xte)[:,1]):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=.8, label="Random baseline")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.legend(frameon=False, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("real_roc.png", dpi=220); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
plot_tree(DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42).fit(Xtr, ytr),
          feature_names=FEATS, class_names=["fail", "success"], filled=True, impurity=False,
          fontsize=7, ax=ax, rounded=True)
fig.tight_layout(); fig.savefig("real_tree.png", dpi=200); plt.close(fig)

cm = confusion_matrix(yte, tree.predict(Xte))
fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=10)
ax.set_xticks([0, 1], ["fail", "success"]); ax.set_yticks([0, 1], ["fail", "success"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.tight_layout(); fig.savefig("real_confusion.png", dpi=220); plt.close(fig)
data.to_csv("oulad_dataset.csv", index=False)
print("\nfigures + oulad_dataset.csv écrits")
