"""
Component 1: Baseline Predictive Models
MSc IT Dissertation - CTI Robustness Project

This component trains three baseline machine learning models for phishing URL detection:
- Logistic Regression (Linear baseline)
- Random Forest (Ensemble method)
- Support Vector Machine with RBF kernel (Non-linear)

Authors: Muhammad Sajid Iqbal, Aneela Shafique, Muhammad Arfan, Muhammad Talha Anwar
Dataset: Kaggle Phishing Websites Dataset (22,110 samples, 30 features)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("COMPONENT 1: BASELINE PREDICTIVE MODELS")
print("="*80)

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('models', exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================
print("\n[1/6] Loading dataset...")
try:
    df = pd.read_csv('Dataset Phising Website.csv')
    print(f"✓ Dataset loaded successfully")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {df.shape[1] - 2} (excluding 'index' and 'Result')")
except FileNotFoundError:
    print("✗ Error: Dataset_Phising_Website.csv not found!")
    print("  Please ensure the dataset is in the same folder as this script.")
    exit(1)

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[2/6] Preprocessing data...")

# Remove index column and prepare features/labels
if 'index' in df.columns:
    df = df.drop('index', axis=1)

X = df.drop('Result', axis=1)
y = df['Result']

# Convert labels to binary (0 and 1)
y_binary = (y == 1).astype(int)

print(f"✓ Features prepared: {X.shape[1]} features")
print(f"  Class distribution:")
print(f"    Legitimate (1): {sum(y_binary == 1)} samples ({sum(y_binary == 1)/len(y_binary)*100:.1f}%)")
print(f"    Phishing (0): {sum(y_binary == 0)} samples ({sum(y_binary == 0)/len(y_binary)*100:.1f}%)")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

print(f"✓ Train-test split complete:")
print(f"    Training set: {len(X_train)} samples")
print(f"    Test set: {len(X_test)} samples")

# Apply SMOTE to training data only (prevent data leakage)
smote = SMOTE(random_state=RANDOM_STATE)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"✓ SMOTE balancing applied:")
print(f"    Balanced training set: {len(X_train_balanced)} samples")
print(f"    Class 0: {sum(y_train_balanced == 0)} samples")
print(f"    Class 1: {sum(y_train_balanced == 1)} samples")

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Feature standardization complete")

# ============================================================================
# STEP 3: TRAIN BASELINE MODELS
# ============================================================================
print("\n[3/6] Training baseline models...")

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=20),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE)
}

results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    model.fit(X_train_scaled, y_train_balanced)
    
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    print(f"    ✓ Accuracy: {results[name]['accuracy']:.4f}")
    print(f"    ✓ Precision: {results[name]['precision']:.4f}")
    print(f"    ✓ Recall: {results[name]['recall']:.4f}")
    print(f"    ✓ F1-Score: {results[name]['f1']:.4f}")

# ============================================================================
# STEP 4: CROSS-VALIDATION
# ============================================================================
print("\n[4/6] Performing 5-fold cross-validation...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=cv, scoring='accuracy')
    cv_results[name] = {
        'mean': scores.mean(),
        'std': scores.std(),
        'scores': scores
    }
    print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")

# ============================================================================
# STEP 5: SAVE RESULTS AND MODELS
# ============================================================================
print("\n[5/6] Saving results and models...")

# Save performance metrics
baseline_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'Precision': [results[m]['precision'] for m in results.keys()],
    'Recall': [results[m]['recall'] for m in results.keys()],
    'F1-Score': [results[m]['f1'] for m in results.keys()],
    'CV Mean': [cv_results[m]['mean'] for m in cv_results.keys()],
    'CV Std': [cv_results[m]['std'] for m in cv_results.keys()]
})

baseline_df.to_csv('results/baseline_results.csv', index=False)
print("  ✓ Saved: results/baseline_results.csv")

# Save trained models
for name, result in results.items():
    model_name = name.replace(' ', '_').replace('(', '').replace(')', '').lower()
    joblib.dump(result['model'], f'models/{model_name}.pkl')
    print(f"  ✓ Saved: models/{model_name}.pkl")

# Save preprocessing objects
joblib.dump(scaler, 'models/scaler.pkl')
print("  ✓ Saved: models/scaler.pkl")

# Save test data for use in other components
test_data = pd.DataFrame(X_test_scaled, columns=X.columns)
test_data['label'] = y_test.values
test_data.to_csv('results/test_data.csv', index=False)
print("  ✓ Saved: results/test_data.csv")

# ============================================================================
# STEP 6: GENERATE VISUALIZATIONS
# ============================================================================
print("\n[6/6] Generating visualizations...")

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (name, result) in enumerate(results.items()):
    cm = result['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Phishing', 'Legitimate'],
                yticklabels=['Phishing', 'Legitimate'],
                cbar=False)
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', 
                       fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('True Label', fontsize=10)
    axes[idx].set_xlabel('Predicted Label', fontsize=10)

plt.tight_layout()
plt.savefig('results/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/confusion_matrices.png")
plt.close()

# 2. ROC Curves
plt.figure(figsize=(10, 8))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_proba'])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curves: Baseline Model Comparison', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/roc_curves.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/roc_curves.png")
plt.close()

# 3. Performance Comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for idx, name in enumerate(results.keys()):
    values = [results[name]['accuracy'], results[name]['precision'], 
              results[name]['recall'], results[name]['f1']]
    ax.bar(x + idx*width, values, width, label=name)

ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Baseline Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.1])
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/performance_comparison.png")
plt.close()

# 4. Cross-Validation Stability
fig, ax = plt.subplots(figsize=(10, 6))
bp = ax.boxplot([cv_results[name]['scores'] for name in cv_results.keys()],
                labels=list(cv_results.keys()),
                patch_artist=True,
                showmeans=True)

colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax.set_ylabel('Cross-Validation Accuracy', fontsize=12, fontweight='bold')
ax.set_title('5-Fold Cross-Validation: Model Stability Analysis', fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.savefig('results/cv_stability.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/cv_stability.png")
plt.close()

# 5. Feature Importance (Random Forest only)
if 'Random Forest' in results:
    rf_model = results['Random Forest']['model']
    feature_names = X.columns
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    
    top_features = [(feature_names[i], importances[i]) for i in indices]
    
    plt.figure(figsize=(10, 6))
    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]
    
    bars = plt.barh(range(len(features)), values, color='steelblue')
    plt.yticks(range(len(features)), features)
    plt.xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
    plt.title('Top 10 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: results/feature_importance.png")
    plt.close()

print("\n" + "="*80)
print("COMPONENT 1 COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  Models:")
print("    - models/logistic_regression.pkl")
print("    - models/random_forest.pkl")
print("    - models/svm_rbf.pkl")
print("    - models/scaler.pkl")
print("\n  Results:")
print("    - results/baseline_results.csv")
print("    - results/test_data.csv")
print("\n  Visualizations:")
print("    - results/confusion_matrices.png")
print("    - results/roc_curves.png")
print("    - results/performance_comparison.png")
print("    - results/cv_stability.png")
print("    - results/feature_importance.png")
print("\nNext: Run component2_adversarial_attacks.py")
print("="*80)
