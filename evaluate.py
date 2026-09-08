"""
evaluate.py -- train and evaluate on the augmented dataset.

Two things this does that a naive script would not:
  * cross-validation folds are grouped by SOURCE BASE CLIP, so augmented
    siblings of the same segment never straddle a fold boundary
  * the FRS comparison covers three variants: none / original (cepstral)
    / corrected (rms)
"""

import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features import create_dataset, COMMANDS

SEED = 42
results = {}


def rule(t):
    print("\n" + "=" * 66); print(f"  {t}"); print("=" * 66)


def load(split, use_frs, frs_mode):
    X, y, s, g = create_dataset(os.path.join("sounds", split), use_frs=use_frs,
                                frs_mode=frs_mode, return_groups=True)
    groups = np.array([f"{a}_{b}" for a, b in zip(y, g)])
    return X, y, groups


def models():
    return {
        "SVM linear (raw)":     SVC(kernel="linear", random_state=SEED),
        "SVM linear (scaled)":  make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=SEED)),
        "SVM rbf (scaled)":     make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)),
        "kNN k=5 (raw)":        KNeighborsClassifier(5),
        "kNN k=5 (scaled)":     make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    }


def main():
    variants = {
        "no FRS (baseline)":       dict(use_frs=False, frs_mode=None),
        "FRS original (cepstral)": dict(use_frs=True,  frs_mode="cepstral"),
        "FRS corrected (rms)":     dict(use_frs=True,  frs_mode="rms"),
    }

    rule("DATASET")
    X0, y0, g0 = load("train", False, None)
    Xt0, yt0, _ = load("test", False, None)
    print(f"\n  train {len(X0)} clips from {len(set(g0))} independent base segments")
    print(f"  test  {len(Xt0)} clips (no augmentation)")
    print(f"  per class (train): {[(COMMANDS[i], int((y0==i).sum())) for i in range(6)]}")
    print(f"  per class (test):  {[(COMMANDS[i], int((yt0==i).sum())) for i in range(6)]}")
    results["n_train"] = int(len(X0)); results["n_test"] = int(len(Xt0))
    results["n_base_segments"] = int(len(set(g0)))

    # ---- 1. grouped CV on the training set -------------------------------
    rule("1. GROUPED CROSS-VALIDATION (folds split by source segment)")
    print(f"\n  {'feature variant':26}{'classifier':22}{'accuracy':>18}")
    print("  " + "-" * 62)
    cv_res = {}
    for vname, kw in variants.items():
        X, y, g = load("train", **kw)
        cv_res[vname] = {}
        for mname, mdl in models().items():
            n_sp = min(5, len(set(g)))
            sc = cross_val_score(mdl, X, y, groups=g, cv=GroupKFold(n_splits=n_sp), n_jobs=-1)
            cv_res[vname][mname] = [round(float(sc.mean()), 4), round(float(sc.std()), 4)]
            print(f"  {vname:26}{mname:22}{sc.mean():>10.1%} +/-{sc.std():>5.1%}")
        print()
    results["grouped_cv"] = cv_res

    # ---- 2. FRS on/off --------------------------------------------------
    rule("2. FRS ON/OFF (grouped CV, SVM linear raw)")
    base = cv_res["no FRS (baseline)"]["SVM linear (raw)"]
    print()
    for vname in variants:
        m, sd = cv_res[vname]["SVM linear (raw)"]
        d = m - base[0]
        print(f"  {vname:26}{m:>8.1%} +/-{sd:>5.1%}   delta vs baseline {d:+.1%}")
    print("\n  (deltas smaller than the +/- are not distinguishable from noise)")

    # ---- 3. held-out test -------------------------------------------------
    rule("3. HELD-OUT TEST SET")
    best_name = max(cv_res["FRS corrected (rms)"], key=lambda k: cv_res["FRS corrected (rms)"][k][0])
    print(f"\n  classifier chosen by CV: {best_name}")
    test_res = {}
    for vname, kw in variants.items():
        X, y, g = load("train", **kw)
        Xt, yt, _ = load("test", **kw)
        mdl = models()[best_name]; mdl.fit(X, y)
        pred = mdl.predict(Xt)
        acc = accuracy_score(yt, pred)
        test_res[vname] = round(float(acc), 4)
        print(f"  {vname:26} test accuracy {acc:>7.1%}  ({int((pred==yt).sum())}/{len(yt)})")
    results["test_accuracy"] = test_res

    X, y, g = load("train", use_frs=True, frs_mode="rms")
    Xt, yt, _ = load("test", use_frs=True, frs_mode="rms")
    mdl = models()[best_name]; mdl.fit(X, y); pred = mdl.predict(Xt)
    print("\n  Per-class report (FRS corrected):")
    # sklearn documents zero_division as {"warn", 0.0, 1.0, np.nan}; its own
    # type stub narrows it to str, so the checker is wrong here, not the call.
    print(classification_report(
        yt, pred, labels=range(6), target_names=COMMANDS,
        zero_division=0))  # pyright: ignore[reportArgumentType]

    cm = confusion_matrix(yt, pred, labels=range(6))
    print("  Confusion matrix (rows = actual, cols = predicted):")
    print(f"  {'':9}" + "".join(f"{c[:7]:>8}" for c in COMMANDS))
    for i, r in enumerate(cm):
        print(f"  {COMMANDS[i]:9}" + "".join(f"{v:>8}" for v in r))
    results["confusion_matrix"] = cm.tolist()

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=COMMANDS,
                yticklabels=COMMANDS, square=True, cbar_kws={"shrink": .8})
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Held-out test set ({len(yt)} clips)")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=200); plt.close()
    print("\n  saved confusion_matrix.png")

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  saved results.json")


if __name__ == "__main__":
    main()
