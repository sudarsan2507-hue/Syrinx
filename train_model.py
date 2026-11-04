"""
train_model.py -- fit the classifier used by listen.py and save it.

Saves a bundle rather than a bare estimator, so the exact feature settings
used at training time travel with the model and the live path cannot drift.
"""

import argparse, os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from features import create_dataset, COMMANDS, SAMPLE_RATE

MODEL_PATH = "vocal_command_model.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-frs", dest="use_frs", action="store_true")
    ap.add_argument("--no-frs", dest="use_frs", action="store_false")
    ap.set_defaults(use_frs=False)
    ap.add_argument("--frs-mode", default="cepstral", choices=["cepstral", "rms"])
    args = ap.parse_args()

    print(f"training with use_frs={args.use_frs} frs_mode={args.frs_mode}")

    Xtr, ytr, _ = create_dataset(os.path.join("sounds", "train"),
                                 use_frs=args.use_frs, frs_mode=args.frs_mode)
    if len(Xtr) == 0:
        print("no training data under sounds/train -- run prepare_data.py then augment.py")
        return

    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(Xtr, ytr)
    print(f"  fitted on {len(Xtr)} clips, {Xtr.shape[1]} features")

    test_dir = os.path.join("sounds", "test")
    if os.path.isdir(test_dir):
        Xte, yte, _ = create_dataset(test_dir, use_frs=args.use_frs, frs_mode=args.frs_mode)
        if len(Xte):
            pred = model.predict(Xte)
            print(f"  held-out test accuracy: {accuracy_score(yte, pred):.1%} "
                  f"({int((pred == yte).sum())}/{len(yte)})")

    bundle = {
        "model": model,
        "commands": COMMANDS,
        "use_frs": args.use_frs,
        "frs_mode": args.frs_mode,
        "sample_rate": SAMPLE_RATE,
        "n_features": int(Xtr.shape[1]),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"  saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
