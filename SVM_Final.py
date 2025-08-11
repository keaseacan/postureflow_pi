#!/usr/bin/env python3
# SVM_final_weights.py — SVM trainer + ALWAYS-on IMF weightages (Python 3.9+)

import argparse, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

print("[running]", __file__)

# ----------------- helpers -----------------
def parse_args():
    p = argparse.ArgumentParser(description="SVM posture classifier + IMF weightages.")
    p.add_argument("--excel", default="breathing_features3_3.xlsx", help="Excel file")
    p.add_argument("--test", type=float, default=0.20, help="Test fraction (0,1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no-person", action="store_true", help="Exclude Person column if present")
    p.add_argument("--c", type=float, default=8.0, help="SVM C")
    p.add_argument("--gamma", default="scale", help="SVM gamma ('scale' or float)")
    p.add_argument("--class-weight", choices=["none","balanced"], default="none")
    p.add_argument("--repeats", type=int, default=30, help="Permutation importance repeats")
    p.add_argument("--save-model", default="svm_model_final.joblib")
    p.add_argument("--save-encoder", default="label_encoder.joblib")
    p.add_argument("--save-weights", default=None, help="CSV path to save IMF weights")
    p.add_argument("--no-plot", action="store_true", help="Disable confusion-matrix plot")
    p.add_argument("--debug", action="store_true", help="Print column diagnostics")
    return p.parse_args()

def find_label_col(df):
    for c in df.columns:
        if str(c).strip().lower() == "position":
            return c
    raise SystemExit("[error] Could not find a 'Position' column.")

def find_imf_cols(df):
    # Accept IMF_1, IMF1, IMF 1, imf_1, etc.; require 1..9
    pat = re.compile(r"^imf[\s_]?(\d+)$", re.IGNORECASE)
    matches = []
    for c in df.columns:
        m = pat.match(str(c).strip())
        if m:
            k = int(m.group(1))
            if 1 <= k <= 9:
                matches.append((k, c))
    if len(matches) < 9:
        found = sorted([c for _, c in matches])
        raise SystemExit(f"[error] Need IMF 1..9 columns; found {len(matches)}: {found}")
    matches.sort(key=lambda x: x[0])
    return [c for _, c in matches]

def find_person_col(df):
    for c in df.columns:
        if "person" in str(c).lower():
            return c
    return None

# ----------------- main -----------------
def main():
    args = parse_args()

    # Load
    try:
        df = pd.read_excel(args.excel, engine="openpyxl")
    except Exception as e:
        raise SystemExit(f"[error] Failed to read Excel '{args.excel}': {e}")

    if args.debug:
        print("[debug] Columns:", list(df.columns))
        print("[debug] Head:\n", df.head(3))

    # Columns
    label_col = find_label_col(df)
    imf_cols  = find_imf_cols(df)
    person_col = None if args.no_person else find_person_col(df)

    if args.debug:
        print(f"[debug] label_col = {label_col}")
        print(f"[debug] imf_cols  = {imf_cols}")
        print(f"[debug] person_col= {person_col}")

    # Features
    X_imf = df[imf_cols].to_numpy()
    if person_col:
        X_person = pd.get_dummies(df[person_col].astype(str), prefix="Person").to_numpy()
        X = np.concatenate([X_imf, X_person], axis=1)
    else:
        X = X_imf

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df[label_col].astype(str).values)

    # Split
    if not (0.0 < args.test < 1.0):
        raise SystemExit("[error] --test must be in (0,1).")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test, stratify=y, random_state=args.seed
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test, random_state=args.seed
        )

    print(f"[info] Split — train:{len(X_train)}  test:{len(X_test)}  (test={args.test:.0%})")

    # SVM pipeline
    class_weight = None if args.class_weight == "none" else "balanced"
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        gamma_val = args.gamma  # 'scale' or 'auto'

    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel="rbf",
            C=float(args.c),
            gamma=gamma_val,
            class_weight=class_weight,
            probability=False,
            random_state=args.seed
        )),
    ])

    # Train
    svm.fit(X_train, y_train)

    # Evaluate
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🏁 Final SVM — Test Acc: {acc:.3f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

    # ---------- IMF weightages (permutation importance) ----------
    try:
        perm = permutation_importance(
            estimator=svm,
            X=X_test, y=y_test,
            scoring="f1_macro",
            n_repeats=args.repeats,
            random_state=args.seed, n_jobs=-1
        )
        # IMF are first 9 features (X = [IMF_1..IMF_9, +optional person dummies])
        imps = np.maximum(perm.importances_mean, 0.0)
        k = min(9, imps.shape[0])
        imf_imps = imps[:k]
        total = imf_imps.sum()
        weights = (imf_imps / total) if total > 0 else np.ones(k) / k

        names = [f"IMF_{i}" for i in range(1, k+1)]
        print("\n== IMF weightages (permutation importance, macro-F1) — IMF order ==")
        for name, w in zip(names, weights):   # <-- no sorting, preserves IMF_1..IMF_9
            print(f"{name}: {w*100:5.1f}%")

        if args.save_weights:
            out = pd.DataFrame({"feature": names, "weight": weights})
            out["weight_percent"] = out["weight"] * 100.0
            # out = out.sort_values("feature")  # (optional) force IMF_1..IMF_9 order
            out.to_csv(args.save_weights, index=False)
            print(f"[info] Saved IMF weightages to {args.save_weights}")

    except Exception as e:
        print(f"[warn] Permutation importance failed: {e}")

    # Quick outcome summary
    unique_pred, counts_pred = np.unique(y_pred, return_counts=True)
    print("\n== Prediction counts on TEST ==")
    for lab, cnt in zip(le.inverse_transform(unique_pred), counts_pred):
        print(f"{lab:>9}: {cnt}")
    print(f"\n[summary] Test Acc={acc:.3f}")

    # Plot (after weights so it never blocks weight printing)
    if not args.no_plot:
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_, yticklabels=le.classes_, cmap="Purples")
        plt.title("Final SVM — Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.show()

    # Save artifacts
    try:
        dump(svm, args.save_model)
        dump(le,  args.save_encoder)
        print(f"[info] Saved {args.save_model} and {args.save_encoder}")
    except Exception as e:
        print(f"[warn] Could not save model/encoder: {e}")

if __name__ == "__main__":
    main()
