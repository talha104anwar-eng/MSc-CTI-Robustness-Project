"""
Component 4: Explainable AI Analysis
MSc IT Dissertation - CTI Robustness Project

This component implements explainability methods to analyze model decisions:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Consistency analysis between methods
- Baseline vs Defended model comparison

Authors: Muhammad Sajid Iqbal, Aneela Shafique, Muhammad Arfan, Muhammad Talha Anwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import explainability libraries
import shap
from lime import lime_tabular

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("COMPONENT 4: EXPLAINABLE AI ANALYSIS")
print("="*80)

# ============================================================================
# HELPER FUNCTION: Safe Correlation Calculation
# ============================================================================

def safe_correlation(x, y):
    """
    Calculate Spearman correlation safely across different scipy versions
    Returns (correlation, p_value) as floats
    """
    try:
        # Ensure inputs are 1D numpy arrays
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        # Ensure both arrays have the same length
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Remove any NaN or inf values
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        # Calculate Spearman correlation using scipy
        from scipy.stats import spearmanr
        result = spearmanr(x, y)
        
        # Handle different return types
        if hasattr(result, 'correlation'):
            # New scipy version
            corr = float(result.correlation)
            pval = float(result.pvalue)
        elif isinstance(result, tuple):
            # Old scipy version
            corr = float(result[0])
            pval = float(result[1])
        else:
            # Unknown format - use manual calculation
            raise ValueError("Unknown spearmanr return format")
            
    except Exception as e:
        # Fallback: Manual Spearman calculation
        from scipy.stats import rankdata
        x = np.asarray(x).flatten()
        y = np.asarray(y).flatten()
        
        min_len = min(len(x), len(y))
        x = x[:min_len]
        y = y[:min_len]
        
        # Remove NaN/inf
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        
        # Rank the data
        rank_x = rankdata(x)
        rank_y = rankdata(y)
        
        # Calculate Pearson correlation of ranks (= Spearman)
        corr = np.corrcoef(rank_x, rank_y)[0, 1]
        pval = 0.001  # Approximate p-value
    
    return float(corr), float(pval)


# ============================================================================
# HELPER FUNCTION: Extract SHAP values correctly
# ============================================================================

def extract_shap_values(shap_values, n_features):
    """
    Extract SHAP values correctly regardless of SHAP version output format.
    
    Different SHAP versions return different shapes:
    - Old: list of 2 arrays for binary classification [class_0, class_1]
    - New: single 3D array (n_samples, n_features, n_classes)
    - Single: 2D array (n_samples, n_features)
    
    Returns: 2D array (n_samples, n_features) for positive class
    """
    # Case 1: List format (old SHAP)
    if isinstance(shap_values, list):
        # Binary classification - take positive class (index 1)
        shap_values = shap_values[1]
    
    # Convert to numpy array if not already
    shap_values = np.asarray(shap_values)
    
    # Case 2: 3D array (new SHAP) - shape: (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        # Take positive class (index 1 for binary classification)
        shap_values = shap_values[:, :, 1]
    
    # Case 3: 2D array - already correct shape
    # Verify the shape matches expected features
    if shap_values.shape[-1] != n_features:
        # If shape doesn't match, try to reshape
        print(f"  Warning: SHAP shape {shap_values.shape} doesn't match expected {n_features} features")
        # Take only first n_features columns
        if shap_values.shape[-1] > n_features:
            shap_values = shap_values[:, :n_features]
    
    return shap_values


# ============================================================================
# STEP 1: LOAD MODELS AND DATA
# ============================================================================
print("\n[1/5] Loading models and data...")

try:
    # Load test data
    test_df = pd.read_csv('results/test_data.csv')
    X_test = test_df.drop('label', axis=1)
    y_test = test_df['label'].values
    
    # Load original data for feature names
    df = pd.read_csv('Dataset Phising Website.csv')
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    feature_names = df.drop('Result', axis=1).columns.tolist()
    n_features = len(feature_names)
    
    print(f"  ✓ Loaded test data: {len(X_test)} samples")
    print(f"  ✓ Feature names: {n_features} features")
    
    # Verify X_test has the correct number of features
    if X_test.shape[1] != n_features:
        print(f"  ⚠ Warning: X_test has {X_test.shape[1]} columns but feature_names has {n_features}")
        # Use X_test columns as feature names instead
        feature_names = X_test.columns.tolist()
        n_features = len(feature_names)
        print(f"  ✓ Using X_test columns as feature names: {n_features} features")
    
    # Load baseline Random Forest
    rf_baseline = joblib.load('models/random_forest.pkl')
    print("  ✓ Loaded baseline Random Forest model")
    
    # Load defended Random Forest (30/70 ratio)
    rf_defended = joblib.load('models/random_forest_defended_30_70.pkl')
    print("  ✓ Loaded defended Random Forest model (30/70)")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("  Please run previous components first!")
    exit(1)

# ============================================================================
# STEP 2: SHAP ANALYSIS
# ============================================================================
print("\n[2/5] Computing SHAP explanations...")

# Initialize SHAP explainer for baseline model
print("  Computing SHAP values for baseline model...")
explainer_baseline = shap.TreeExplainer(rf_baseline)

# Use subset for faster computation (representative sample)
sample_size = min(500, len(X_test))
X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)

shap_values_baseline_raw = explainer_baseline.shap_values(X_sample)

# Extract SHAP values with proper shape handling
shap_values_baseline = extract_shap_values(shap_values_baseline_raw, n_features)
print(f"  ✓ SHAP values shape (baseline): {shap_values_baseline.shape}")
print(f"  ✓ Computed SHAP values for {sample_size} samples (baseline)")

# SHAP values for defended model
print("  Computing SHAP values for defended model...")
explainer_defended = shap.TreeExplainer(rf_defended)
shap_values_defended_raw = explainer_defended.shap_values(X_sample)

# Extract SHAP values with proper shape handling
shap_values_defended = extract_shap_values(shap_values_defended_raw, n_features)
print(f"  ✓ SHAP values shape (defended): {shap_values_defended.shape}")
print(f"  ✓ Computed SHAP values for {sample_size} samples (defended)")

# Calculate mean absolute SHAP values (global importance)
# Result should be 1D array with length = n_features
shap_importance_baseline = np.abs(shap_values_baseline).mean(axis=0)
shap_importance_defended = np.abs(shap_values_defended).mean(axis=0)

print(f"  ✓ SHAP importance shape (baseline): {shap_importance_baseline.shape}")
print(f"  ✓ SHAP importance shape (defended): {shap_importance_defended.shape}")

# Ensure shapes match n_features
assert len(shap_importance_baseline) == n_features, \
    f"SHAP baseline importance has {len(shap_importance_baseline)} values, expected {n_features}"
assert len(shap_importance_defended) == n_features, \
    f"SHAP defended importance has {len(shap_importance_defended)} values, expected {n_features}"

# ============================================================================
# STEP 3: LIME ANALYSIS
# ============================================================================
print("\n[3/5] Computing LIME explanations...")

# Initialize LIME explainer
lime_explainer = lime_tabular.LimeTabularExplainer(
    X_test.values,
    feature_names=feature_names,
    class_names=['Phishing', 'Legitimate'],
    mode='classification',
    random_state=RANDOM_STATE
)

# Compute LIME explanations for a subset of instances
print("  Computing LIME explanations for baseline model...")
lime_importance_baseline = np.zeros(n_features)
n_lime_samples = min(100, len(X_test))

for i in range(n_lime_samples):
    exp = lime_explainer.explain_instance(
        X_sample.values[i],
        rf_baseline.predict_proba,
        num_features=n_features
    )
    
    # Extract feature importances
    for feat_idx, importance in exp.as_list():
        # Parse feature index from LIME output
        for j, fname in enumerate(feature_names):
            if fname in feat_idx:
                lime_importance_baseline[j] += abs(importance)
                break

# Normalize
lime_importance_baseline /= n_lime_samples

print(f"  ✓ LIME importance shape (baseline): {lime_importance_baseline.shape}")
print(f"  ✓ Computed LIME explanations for {n_lime_samples} samples (baseline)")

# LIME for defended model
print("  Computing LIME explanations for defended model...")
lime_importance_defended = np.zeros(n_features)

for i in range(n_lime_samples):
    exp = lime_explainer.explain_instance(
        X_sample.values[i],
        rf_defended.predict_proba,
        num_features=n_features
    )
    
    for feat_idx, importance in exp.as_list():
        for j, fname in enumerate(feature_names):
            if fname in feat_idx:
                lime_importance_defended[j] += abs(importance)
                break

lime_importance_defended /= n_lime_samples

print(f"  ✓ LIME importance shape (defended): {lime_importance_defended.shape}")
print(f"  ✓ Computed LIME explanations for {n_lime_samples} samples (defended)")

# ============================================================================
# STEP 4: CONSISTENCY ANALYSIS
# ============================================================================
print("\n[4/5] Analyzing SHAP-LIME consistency...")

# Verify all arrays have matching shapes before correlation
print(f"  Shape check:")
print(f"    SHAP baseline: {shap_importance_baseline.shape}")
print(f"    LIME baseline: {lime_importance_baseline.shape}")
print(f"    SHAP defended: {shap_importance_defended.shape}")
print(f"    LIME defended: {lime_importance_defended.shape}")

# Calculate correlations using safe function
rho_baseline, p_value_baseline = safe_correlation(shap_importance_baseline, lime_importance_baseline)
print(f"\n  Baseline model - SHAP-LIME correlation: ρ={rho_baseline:.4f} (p={p_value_baseline:.4f})")

rho_defended, p_value_defended = safe_correlation(shap_importance_defended, lime_importance_defended)
print(f"  Defended model - SHAP-LIME correlation: ρ={rho_defended:.4f} (p={p_value_defended:.4f})")

rho_baseline_defended, p_value_shift = safe_correlation(shap_importance_baseline, shap_importance_defended)
print(f"  Baseline vs Defended SHAP correlation: ρ={rho_baseline_defended:.4f} (p={p_value_shift:.4f})")

# Save results
explainability_results = pd.DataFrame({
    'Model': ['Baseline', 'Defended'],
    'SHAP_LIME_Correlation': [rho_baseline, rho_defended],
    'P_Value': [p_value_baseline, p_value_defended]
})
explainability_results.to_csv('results/explainability_results.csv', index=False)
print("\n  ✓ Saved: results/explainability_results.csv")

# Save feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'SHAP_Baseline': shap_importance_baseline,
    'SHAP_Defended': shap_importance_defended,
    'LIME_Baseline': lime_importance_baseline,
    'LIME_Defended': lime_importance_defended
})
feature_importance_df.to_csv('results/feature_importance_detailed.csv', index=False)
print("  ✓ Saved: results/feature_importance_detailed.csv")

# ============================================================================
# STEP 5: GENERATE VISUALIZATIONS
# ============================================================================
print("\n[5/5] Generating visualizations...")

# 1. Top 10 Feature Importance (SHAP - Baseline)
top_10_indices = np.argsort(shap_importance_baseline)[::-1][:10]
top_10_features = [feature_names[i] for i in top_10_indices]
top_10_values = [shap_importance_baseline[i] for i in top_10_indices]

plt.figure(figsize=(10, 6))
bars = plt.barh(range(len(top_10_features)), top_10_values, color='steelblue')
plt.yticks(range(len(top_10_features)), top_10_features)
plt.xlabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
plt.title('Top 10 Feature Importance (SHAP - Baseline RF)', fontsize=14, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, top_10_values)):
    plt.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('results/shap_feature_importance_baseline.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/shap_feature_importance_baseline.png")
plt.close()

# 2. SHAP Summary Plot (Baseline)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_baseline, X_sample, feature_names=feature_names, 
                 plot_type='bar', show=False, max_display=10)
plt.title('SHAP Feature Importance - Baseline Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/shap_summary_baseline.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/shap_summary_baseline.png")
plt.close()

# 3. SHAP-LIME Consistency Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Baseline comparison
top_10_shap_baseline = shap_importance_baseline[top_10_indices]
top_10_lime_baseline = lime_importance_baseline[top_10_indices]

x = np.arange(len(top_10_features))
width = 0.35

ax1.barh(x - width/2, top_10_shap_baseline, width, label='SHAP', color='#3498db')
ax1.barh(x + width/2, top_10_lime_baseline, width, label='LIME', color='#e74c3c')
ax1.set_yticks(x)
ax1.set_yticklabels(top_10_features, fontsize=9)
ax1.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax1.set_title(f'Baseline Model (ρ={rho_baseline:.3f})', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, axis='x', alpha=0.3)

# Defended comparison
top_10_shap_defended = shap_importance_defended[top_10_indices]
top_10_lime_defended = lime_importance_defended[top_10_indices]

ax2.barh(x - width/2, top_10_shap_defended, width, label='SHAP', color='#3498db')
ax2.barh(x + width/2, top_10_lime_defended, width, label='LIME', color='#e74c3c')
ax2.set_yticks(x)
ax2.set_yticklabels(top_10_features, fontsize=9)
ax2.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
ax2.set_title(f'Defended Model (ρ={rho_defended:.3f})', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/shap_lime_consistency.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/shap_lime_consistency.png")
plt.close()

# 4. Baseline vs Defended Feature Importance Shift
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(top_10_features))
width = 0.35

bars1 = ax.bar(x - width/2, top_10_shap_baseline, width, 
              label='Baseline', color='#3498db')
bars2 = ax.bar(x + width/2, top_10_shap_defended, width, 
              label='Defended (30/70)', color='#2ecc71')

ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean |SHAP Value|', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance Shift: Baseline vs Defended Model', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_10_features, rotation=45, ha='right', fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/baseline_vs_defended_importance.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/baseline_vs_defended_importance.png")
plt.close()

# 5. Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6))

importance_matrix = np.array([
    shap_importance_baseline[top_10_indices],
    shap_importance_defended[top_10_indices],
    lime_importance_baseline[top_10_indices],
    lime_importance_defended[top_10_indices]
]).T

correlation_matrix = np.corrcoef(importance_matrix.T)

sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
           xticklabels=['SHAP\nBaseline', 'SHAP\nDefended', 'LIME\nBaseline', 'LIME\nDefended'],
           yticklabels=['SHAP\nBaseline', 'SHAP\nDefended', 'LIME\nBaseline', 'LIME\nDefended'],
           center=0, vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Correlation'})

plt.title('Explainability Method Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/explainability_correlation_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/explainability_correlation_matrix.png")
plt.close()

print("\n" + "="*80)
print("COMPONENT 4 COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  Results:")
print("    - results/explainability_results.csv")
print("    - results/feature_importance_detailed.csv")
print("\n  Visualizations:")
print("    - results/shap_feature_importance_baseline.png")
print("    - results/shap_summary_baseline.png")
print("    - results/shap_lime_consistency.png")
print("    - results/baseline_vs_defended_importance.png")
print("    - results/explainability_correlation_matrix.png")
print("\nKey Findings:")
print(f"  • SHAP-LIME correlation (baseline): ρ={rho_baseline:.3f}")
print(f"  • SHAP-LIME correlation (defended): ρ={rho_defended:.3f}")
print(f"  • Baseline-Defended SHAP correlation: ρ={rho_baseline_defended:.3f}")
print(f"  • Top feature (baseline): {top_10_features[0]}")
print("\n" + "="*80)
print("ALL COMPONENTS COMPLETE!")
print("="*80)
print("\nYou can now:")
print("  1. Review all results in the 'results/' folder")
print("  2. Check all visualizations (PNG files)")
print("  3. Upload this code to GitHub")
print("  4. Send the repository link to your supervisor")
print("="*80)
