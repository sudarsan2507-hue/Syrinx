"""
train.py — train the classifier and report how well it does.

Runs in one pass:
  1. learning curve over dataset sizes (30 / 120 / 180)
  2. FRS vs no-FRS comparison at full size
  3. SVM vs kNN on an identical split
  4. held-out test evaluation
  5. cross-speaker evaluation if more than one speaker is present
  6. confusion matrix saved as confusion_matrix.png

A summary of everything is printed at the end.

Usage:  python train.py
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from features import create_dataset, COMMANDS

SEED = 42
TRAIN_DIR = os.path.join("sounds", "train")
TEST_DIR = os.path.join("sounds", "test")

results = {}


def rule(title):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def stratified_subsample(X, y, per_class, seed=SEED):
    """Take `per_class` samples from each class, preserving balance."""
    rng = np.random.RandomState(seed)
    idx = []
    for label in range(len(COMMANDS)):
        pool = np.where(y == label)[0]
        take = min(per_class, len(pool))
        idx.extend(rng.choice(pool, size=take, replace=False))
    idx = np.array(sorted(idx))
    return X[idx], y[idx]


def fit_svm(X_tr, y_tr):
    model = SVC(kernel="linear", probability=True, random_state=SEED)
    model.fit(X_tr, y_tr)
    return model


def evaluate(model, X_tr, y_tr, X_val, y_val):
    train_acc = accuracy_score(y_tr, model.predict(X_tr))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    return train_acc, val_acc


def main():
    rule("LOADING TRAINING DATA (FRS enabled)")
    X, y, spk = create_dataset(TRAIN_DIR, use_frs=True, frs_mode="cepstral")

    if len(X) == 0:
        print("\nNo training data found. Run record.py first.")
        return

    print(f"\n  samples: {len(X)}   feature dim: {X.shape[1]}")
    counts = {COMMANDS[i]: int(np.sum(y == i)) for i in range(len(COMMANDS))}
    print(f"  per class: {counts}")
    speakers = sorted(set(spk))
    print(f"  speakers: {speakers}")

    min_per_class = min(counts.values())
    results["n_total"] = int(len(X))
    results["per_class"] = counts
    results["speakers"] = speakers

    # ---------------------------------------------------------------
    # 1. Learning curve
    # ---------------------------------------------------------------
    rule("1. LEARNING CURVE (SVM, stratified 75/25 split)")
    print(f"\n  {'total':>6}  {'per class':>9}  {'train acc':>9}  {'val acc':>8}  {'gap':>6}")
    print("  " + "-" * 48)

    curve = []
    for per_class in (5, 20, 30):
        total = per_class * len(COMMANDS)
        if per_class > min_per_class:
            print(f"  {total:>6}  {per_class:>9}   -- not enough data --")
            continue

        Xs, ys = stratified_subsample(X, y, per_class)
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xs, ys, test_size=0.25, random_state=SEED, stratify=ys
        )
        model = fit_svm(X_tr, y_tr)
        tr, va = evaluate(model, X_tr, y_tr, X_val, y_val)
        gap = tr - va
        print(f"  {total:>6}  {per_class:>9}  {tr:>8.1%}  {va:>7.1%}  {gap:>5.1%}")
        curve.append(
            {"total": total, "per_class": per_class,
             "train_acc": round(tr, 4), "val_acc": round(va, 4),
             "gap": round(gap, 4)}
        )

    results["learning_curve"] = curve

    # ---------------------------------------------------------------
    # 2. FRS on/off
    # ---------------------------------------------------------------
    rule("2. FRS ON/OFF (full dataset, identical split)")

    X_nofrs, y_nofrs, _ = create_dataset(TRAIN_DIR, use_frs=False, frs_mode="cepstral")

    frs_compare = {}
    for tag, (Xa, ya) in {
        "with_frs": (X, y),
        "without_frs": (X_nofrs, y_nofrs),
    }.items():
        X_tr, X_val, y_tr, y_val = train_test_split(
            Xa, ya, test_size=0.25, random_state=SEED, stratify=ya
        )
        model = fit_svm(X_tr, y_tr)
        tr, va = evaluate(model, X_tr, y_tr, X_val, y_val)
        frs_compare[tag] = {"train_acc": round(tr, 4), "val_acc": round(va, 4)}
        label = "with FRS" if tag == "with_frs" else "without FRS"
        print(f"\n  {label:14}  train {tr:.1%}   val {va:.1%}")

    delta = frs_compare["with_frs"]["val_acc"] - frs_compare["without_frs"]["val_acc"]
    frs_compare["delta_val"] = round(delta, 4)
    print(f"\n  FRS effect on validation accuracy: {delta:+.1%}")
    results["frs_compare"] = frs_compare

    # ---------------------------------------------------------------
    # 3. SVM vs kNN, same split
    # ---------------------------------------------------------------
    rule("3. CLASSIFIER COMPARISON (identical split, FRS features)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    svm = fit_svm(X_tr, y_tr)
    svm_tr, svm_va = evaluate(svm, X_tr, y_tr, X_val, y_val)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_tr)
    knn_tr, knn_va = evaluate(knn, X_tr, y_tr, X_val, y_val)

    print(f"\n  {'classifier':<14}{'train':>9}{'val':>9}")
    print("  " + "-" * 32)
    print(f"  {'SVM (linear)':<14}{svm_tr:>8.1%}{svm_va:>9.1%}")
    print(f"  {'kNN (k=5)':<14}{knn_tr:>8.1%}{knn_va:>9.1%}")

    results["classifier_comparison"] = {
        "svm": {"train_acc": round(svm_tr, 4), "val_acc": round(svm_va, 4)},
        "knn": {"train_acc": round(knn_tr, 4), "val_acc": round(knn_va, 4)},
    }

    print("\n  Validation classification report (SVM):")
    val_pred = svm.predict(X_val)
    print(classification_report(
        y_val, val_pred, labels=range(len(COMMANDS)),
        target_names=COMMANDS,
        zero_division=0))  # pyright: ignore[reportArgumentType]

    # ---------------------------------------------------------------
    # 4. Held-out test set
    # ---------------------------------------------------------------
    rule("4. HELD-OUT TEST SET")

    X_test, y_test, spk_test = create_dataset(TEST_DIR, use_frs=True, frs_mode="cepstral")

    if len(X_test) == 0:
        print("\n  No test data found — skipping.")
        print("  Record a test set with record.py in a separate session.")
        results["test"] = None
    else:
        final = fit_svm(X, y)          # train on everything for the final model
        test_pred = final.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)

        print(f"\n  test samples: {len(X_test)}")
        print(f"  TEST ACCURACY: {test_acc:.1%}")
        print("\n  Test classification report:")
        print(classification_report(
            y_test, test_pred, labels=range(len(COMMANDS)),
            target_names=COMMANDS,
            zero_division=0))  # pyright: ignore[reportArgumentType]

        cm = confusion_matrix(y_test, test_pred, labels=range(len(COMMANDS)))
        print("  Confusion matrix (rows = actual, cols = predicted):")
        print(f"  {'':10}" + "".join(f"{c[:6]:>8}" for c in COMMANDS))
        for i, row in enumerate(cm):
            print(f"  {COMMANDS[i]:10}" + "".join(f"{v:>8}" for v in row))

        plt.figure(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=COMMANDS, yticklabels=COMMANDS,
            cbar_kws={"shrink": 0.8}, square=True,
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion matrix, held-out test set")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("\n  saved confusion_matrix.png")

        joblib.dump(final, "vocal_command_model.pkl")
        print("  saved vocal_command_model.pkl")

        results["test"] = {
            "n": int(len(X_test)),
            "accuracy": round(test_acc, 4),
            "confusion_matrix": cm.tolist(),
        }

    # ---------------------------------------------------------------
    # 5. Cross-speaker (leave-one-speaker-out)
    # ---------------------------------------------------------------
    if len(speakers) > 1:
        rule("5. CROSS-SPEAKER (leave one speaker out)")
        loso = {}
        for held in speakers:
            tr_mask = spk != held
            te_mask = spk == held
            if te_mask.sum() == 0 or len(set(y[tr_mask])) < len(COMMANDS):
                continue
            model = fit_svm(X[tr_mask], y[tr_mask])
            acc = accuracy_score(y[te_mask], model.predict(X[te_mask]))
            loso[held] = round(acc, 4)
            print(f"  held out {held}: {acc:.1%}  ({int(te_mask.sum())} samples)")
        if loso:
            mean_acc = float(np.mean(list(loso.values())))
            print(f"\n  mean cross-speaker accuracy: {mean_acc:.1%}")
            results["cross_speaker"] = {"per_speaker": loso, "mean": round(mean_acc, 4)}
    else:
        print("\n  (only one speaker — cross-speaker evaluation skipped)")
        results["cross_speaker"] = None

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    rule("SUMMARY")
    print(json.dumps(results, indent=2))

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n  also saved to results.json")


if __name__ == "__main__":
    main()