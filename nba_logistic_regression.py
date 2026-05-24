"""
NBA Rookie Career Longevity Predictor - Logistic Regression Pipeline
Author: Antigravity AI
Description: A comprehensive, 12-step machine learning pipeline that loads NBA rookie data,
             preprocesses features, trains a baseline Logistic Regression, interprets feature coefficients,
             tunes hyperparameters (L1/L2 penalties & regularization strength C), and benchmarks the models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve
)

# Set custom styling for premium visualizations
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

# Custom premium palette
PRIMARY_COLOR = "#2C3E50"  # Midnight Blue
ACCENT_COLOR = "#E74C3C"   # Coral Red
SUCCESS_COLOR = "#2ECC71"  # Emerald Green
NEUTRAL_LIGHT = "#ECF0F1"  # Cool Light Grey

def main():
    print("=" * 80)
    print("        STARTING NBA ROOKIE CAREER LONGEVITY LOGISTIC REGRESSION PIPELINE")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 1: Load Dataset
    # -------------------------------------------------------------------------
    print("\n--- STEP 1: Loading Dataset ---")
    csv_path = "nba_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing '{csv_path}' in the workspace directory.")
    
    df = pd.read_csv(csv_path)
    print(f"Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    print("Columns available:", list(df.columns))
    
    # -------------------------------------------------------------------------
    # STEP 2: Clean Missing Values
    # -------------------------------------------------------------------------
    print("\n--- STEP 2: Cleaning Missing Values ---")
    missing_counts = df.isnull().sum()
    columns_with_missing = missing_counts[missing_counts > 0]
    
    if len(columns_with_missing) > 0:
        print("Missing values detected:")
        for col, count in columns_with_missing.items():
            print(f"  - {col}: {count} missing values ({count / len(df) * 100:.2f}%)")
    else:
        print("No missing values detected in the raw CSV!")

    # Separate target '5Yrs' and features
    target_col = "5Yrs"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    X_raw = df.drop(columns=[target_col])
    y = df[target_col]

    # Clean numerical missing values using Median Imputation
    num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    
    # Impute missing values for numeric columns
    imputer = SimpleImputer(strategy="median")
    X_imputed = X_raw.copy()
    X_imputed[num_cols] = imputer.fit_transform(X_raw[num_cols])
    print(f"Median imputation successfully completed for numerical features.")

    # -------------------------------------------------------------------------
    # STEP 3: Remove/Encode Text Features
    # -------------------------------------------------------------------------
    print("\n--- STEP 3: Removing/Encoding Text Features ---")
    # Identify non-numeric columns (usually player Name)
    text_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"Non-numeric columns identified: {text_cols}")
    
    # Save Name column as metadata for tracking individual predictions
    player_names = None
    if "Name" in text_cols:
        player_names = X_imputed["Name"]
        X_features = X_imputed.drop(columns=["Name"])
        print("Dropped 'Name' column from training features and saved it as tracking metadata.")
    else:
        X_features = X_imputed
        print("No 'Name' column found. Features remain as-is.")
        
    feature_names = X_features.columns.tolist()
    print(f"Final feature count for training: {len(feature_names)}")
    print("Features list:", feature_names)

    # -------------------------------------------------------------------------
    # STEP 4: Feature Scaling
    # -------------------------------------------------------------------------
    print("\n--- STEP 4: Feature Scaling ---")
    # Use StandardScaler to standardize numerical columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)
    print("Applied StandardScaler: features have mean = 0.0 and variance = 1.0.")

    # -------------------------------------------------------------------------
    # STEP 5: Train-Test Split
    # -------------------------------------------------------------------------
    print("\n--- STEP 5: Train-Test Split ---")
    # Stratified 80-20 split to maintain the class balance of the target '5Yrs'
    X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
        X_scaled, y, player_names, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Splitting completed (Stratified 80-20, Random State = 42):")
    print(f"  - Train set shape: {X_train.shape[0]} rows")
    print(f"  - Test set shape: {X_test.shape[0]} rows")
    print(f"  - Class distribution in Train set: {np.bincount(y_train)}")
    print(f"  - Class distribution in Test set: {np.bincount(y_test)}")

    # -------------------------------------------------------------------------
    # STEP 6: Train Baseline Logistic Regression
    # -------------------------------------------------------------------------
    print("\n--- STEP 6: Training Baseline Logistic Regression ---")
    # Baseline: L2 penalty (Ridge), C=1.0, solver='liblinear'
    baseline_model = LogisticRegression(solver="liblinear", penalty="l2", C=1.0, random_state=42)
    baseline_model.fit(X_train, y_train)
    print("Baseline model trained successfully.")

    # -------------------------------------------------------------------------
    # STEP 7: Predict
    # -------------------------------------------------------------------------
    print("\n--- STEP 7: Generating Predictions ---")
    y_pred_base = baseline_model.predict(X_test)
    y_prob_base = baseline_model.predict_proba(X_test)[:, 1]
    print("Class labels and prediction probabilities successfully generated for the test set.")

    # -------------------------------------------------------------------------
    # STEP 8: Evaluate Baseline Metrics
    # -------------------------------------------------------------------------
    print("\n--- STEP 8: Evaluating Baseline Model Metrics ---")
    
    # Calculate key classification metrics
    base_accuracy = accuracy_score(y_test, y_pred_base)
    base_precision = precision_score(y_test, y_pred_base)
    base_recall = recall_score(y_test, y_pred_base)
    base_f1 = f1_score(y_test, y_pred_base)
    base_auc = roc_auc_score(y_test, y_prob_base)
    
    print("-" * 50)
    print(f"Baseline Accuracy:  {base_accuracy:.4f}")
    print(f"Baseline Precision: {base_precision:.4f}")
    print(f"Baseline Recall:    {base_recall:.4f}")
    print(f"Baseline F1-Score:  {base_f1:.4f}")
    print(f"Baseline ROC-AUC:   {base_auc:.4f}")
    print("-" * 50)
    
    print("\nClassification Report (Baseline Model):")
    print(classification_report(y_test, y_pred_base, target_names=["< 5 Yrs", ">= 5 Yrs"]))

    # -------------------------------------------------------------------------
    # STEP 9: Interpret Coefficients
    # -------------------------------------------------------------------------
    print("\n--- STEP 9: Interpreting Coefficients ---")
    coefficients = baseline_model.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    # Create coefficient interpretation dataframe
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient (Beta)": coefficients,
        "Odds Ratio (exp(Beta))": odds_ratios,
        "Impact Direction": ["Positive" if c > 0 else "Negative" for c in coefficients],
        "Abs Magnitude": np.abs(coefficients)
    }).sort_values(by="Abs Magnitude", ascending=False)
    
    print("\nStandardized Coefficients and Odds Ratios (sorted by absolute magnitude):")
    print(coef_df[["Feature", "Coefficient (Beta)", "Odds Ratio (exp(Beta))", "Impact Direction"]].to_string(index=False))
    
    # Save beautiful feature importance plot
    plt.figure(figsize=(10, 6))
    colors = [SUCCESS_COLOR if c > 0 else ACCENT_COLOR for c in coef_df["Coefficient (Beta)"]]
    
    # Plotting Odds Ratios relative to baseline of 1.0
    sns.barplot(
        x="Coefficient (Beta)", 
        y="Feature", 
        data=coef_df, 
        palette=colors,
        hue="Feature",
        legend=False
    )
    plt.axvline(x=0.0, color=PRIMARY_COLOR, linestyle="--", linewidth=1.5, alpha=0.7)
    plt.title("Logistic Regression Standardized Coefficients (Log-Odds Impact)", pad=15)
    plt.xlabel("Coefficient Value (Standardized Beta)")
    plt.ylabel("Rookie Stat Feature")
    plt.tight_layout()
    
    coef_plot_path = "feature_importance.png"
    plt.savefig(coef_plot_path, dpi=300)
    plt.close()
    print(f"\nCoefficient plot successfully saved to '{coef_plot_path}'")

    # -------------------------------------------------------------------------
    # STEP 10: Tune Regularization
    # -------------------------------------------------------------------------
    print("\n--- STEP 10: Tuning Regularization Strength (GridSearchCV) ---")
    
    # Define hyperparameter grid
    param_grid = {
        "C": np.logspace(-3, 3, 20),           # Logarithmic grid for regularization strength C
        "penalty": ["l1", "l2"]                # L1 (Lasso) vs L2 (Ridge) penalties
    }
    
    # Use Stratified 5-Fold Cross Validation
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=LogisticRegression(solver="liblinear", random_state=42),
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="f1",  # Target F1-Score as optimization metric to handle any class imbalances
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print(f"Optimal parameters found:")
    print(f"  - Regularization Penalty: {best_params['penalty'].upper()}")
    print(f"  - C value (regularization strength Inverse): {best_params['C']:.6f} (Optimal alpha = {1.0/best_params['C']:.6f})")
    
    # Save the tuned model
    tuned_model = grid_search.best_estimator_

    # -------------------------------------------------------------------------
    # STEP 11: Final Baseline Benchmark
    # -------------------------------------------------------------------------
    print("\n--- STEP 11: Final Baseline Benchmark Comparison ---")
    
    # Generate predictions from tuned model
    y_pred_tuned = tuned_model.predict(X_test)
    y_prob_tuned = tuned_model.predict_proba(X_test)[:, 1]
    
    # Calculate tuned metrics
    tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
    tuned_precision = precision_score(y_test, y_pred_tuned)
    tuned_recall = recall_score(y_test, y_pred_tuned)
    tuned_f1 = f1_score(y_test, y_pred_tuned)
    tuned_auc = roc_auc_score(y_test, y_prob_tuned)
    
    # Compile comparison table
    comparison_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        "Baseline Model (L2, C=1.0)": [base_accuracy, base_precision, base_recall, base_f1, base_auc],
        "Tuned Model": [tuned_accuracy, tuned_precision, tuned_recall, tuned_f1, tuned_auc]
    })
    comparison_df["Difference"] = comparison_df["Tuned Model"] - comparison_df["Baseline Model (L2, C=1.0)"]
    
    print("\n" + "="*70)
    print("                       MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(comparison_df.to_string(index=False, formatters={
        "Baseline Model (L2, C=1.0)": "{:.4f}".format,
        "Tuned Model": "{:.4f}".format,
        "Difference": "{:+.4f}".format
    }))
    print("="*70)

    # -------------------------------------------------------------------------
    # STEP 12: Save Final Visualizations
    # -------------------------------------------------------------------------
    print("\n--- STEP 12: Generating Evaluation Visualizations ---")
    
    # 1. Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    cm_base = confusion_matrix(y_test, y_pred_base)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned)
    
    # Baseline Confusion Matrix Heatmap
    sns.heatmap(
        cm_base, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=["< 5 Yrs", ">= 5 Yrs"], yticklabels=["< 5 Yrs", ">= 5 Yrs"]
    )
    axes[0].set_title("Baseline Model Confusion Matrix", pad=10)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")
    
    # Tuned Confusion Matrix Heatmap
    sns.heatmap(
        cm_tuned, annot=True, fmt="d", cmap="Greens", cbar=False, ax=axes[1],
        xticklabels=["< 5 Yrs", ">= 5 Yrs"], yticklabels=["< 5 Yrs", ">= 5 Yrs"]
    )
    axes[1].set_title(f"Tuned Model Confusion Matrix\n({best_params['penalty'].upper()} penalty, C={best_params['C']:.4f})", pad=10)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")
    
    plt.suptitle("Confusion Matrix Comparison", y=1.02)
    plt.tight_layout()
    cm_plot_path = "confusion_matrix.png"
    plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix comparison to '{cm_plot_path}'")

    # 2. ROC Curve Plot
    plt.figure(figsize=(8, 6.5))
    
    # Calculate ROC curve values
    fpr_base, tpr_base, _ = roc_curve(y_test, y_prob_base)
    fpr_tuned, tpr_tuned, _ = roc_curve(y_test, y_prob_tuned)
    
    plt.plot(fpr_base, tpr_base, color=PRIMARY_COLOR, linewidth=2.5, label=f"Baseline Model (AUC = {base_auc:.4f})")
    plt.plot(fpr_tuned, tpr_tuned, color=SUCCESS_COLOR, linewidth=2.5, linestyle="--", label=f"Tuned Model (AUC = {tuned_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#BDC3C7", linestyle=":", linewidth=1.5, label="Random Guess (AUC = 0.5000)")
    
    plt.title("ROC Curve Comparison", pad=15)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none")
    plt.tight_layout()
    
    roc_plot_path = "roc_curve.png"
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve comparison to '{roc_plot_path}'")
    
    print("\n" + "=" * 80)
    print("             PIPELINE COMPLETED SUCCESSFULLY! ALL ARTIFACTS GENERATED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
