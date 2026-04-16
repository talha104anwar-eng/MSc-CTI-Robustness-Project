"""
Component 2: Adversarial Attack Assessment
MSc IT Dissertation - CTI Robustness Project

This component generates adversarial attacks against the baseline models:
- FGSM-like perturbation attacks (compatible with all models)
- PGD-like iterative attacks
- Robustness evaluation across all models

Authors: Muhammad Sajid Iqbal, Aneela Shafique, Muhammad Arfan, Muhammad Talha Anwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("COMPONENT 2: ADVERSARIAL ATTACK ASSESSMENT")
print("="*80)

# ============================================================================
# STEP 1: LOAD TRAINED MODELS AND TEST DATA
# ============================================================================
print("\n[1/5] Loading trained models and test data...")

try:
    # Load models
    lr_model = joblib.load('models/logistic_regression.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    svm_model = joblib.load('models/svm_rbf.pkl')
    
    print("  ✓ Loaded logistic_regression.pkl")
    print("  ✓ Loaded random_forest.pkl")
    print("  ✓ Loaded svm_rbf.pkl")
    
    # Load test data
    test_df = pd.read_csv('results/test_data.csv')
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    print(f"  ✓ Loaded test data: {len(X_test)} samples")
    
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("  Please run component1_baseline_models.py first!")
    exit(1)

# ============================================================================
# STEP 2: GENERATE ADVERSARIAL EXAMPLES USING PERTURBATION
# ============================================================================
print("\n[2/5] Generating adversarial perturbations...")

def generate_fgsm_like_perturbations(X, epsilon):
    """
    Generate FGSM-like perturbations using random noise
    (compatible with all model types including tree-based)
    """
    # Add random perturbation scaled by epsilon
    noise = np.random.normal(0, 1, X.shape)
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)  # Normalize
    X_adv = X + epsilon * noise
    return X_adv

def generate_pgd_like_perturbations(X, epsilon, iterations=10):
    """
    Generate PGD-like perturbations using iterative random noise
    """
    X_adv = X.copy()
    step_size = epsilon / iterations
    
    for _ in range(iterations):
        noise = np.random.normal(0, 1, X.shape)
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        X_adv = X_adv + step_size * noise
        # Project back to epsilon ball
        perturbation = X_adv - X
        perturbation_norm = np.linalg.norm(perturbation, axis=1, keepdims=True)
        perturbation_norm = np.maximum(perturbation_norm, 1e-10)  # Avoid division by zero
        perturbation = perturbation * np.minimum(epsilon / perturbation_norm, 1)
        X_adv = X + perturbation
    
    return X_adv

print("  ✓ Perturbation functions ready")

# ============================================================================
# STEP 3: FGSM-LIKE ATTACKS
# ============================================================================
print("\n[3/5] Generating FGSM-like attacks...")

models_dict = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'SVM (RBF)': svm_model
}

epsilons = [0.01, 0.05, 0.1, 0.2]
fgsm_results = []

for name, model in models_dict.items():
    print(f"\n  Attacking {name}...")
    
    # Get baseline accuracy
    y_pred_clean = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred_clean)
    
    for eps in epsilons:
        # Generate adversarial examples
        X_adv = generate_fgsm_like_perturbations(X_test, eps)
        
        # Evaluate on adversarial examples
        y_pred_adv = model.predict(X_adv)
        adv_acc = accuracy_score(y_test, y_pred_adv)
        
        # Calculate attack success rate
        attack_success_rate = (baseline_acc - adv_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
        
        fgsm_results.append({
            'Model': name,
            'Epsilon': eps,
            'Baseline_Accuracy': baseline_acc,
            'Adversarial_Accuracy': adv_acc,
            'Accuracy_Drop': baseline_acc - adv_acc,
            'Attack_Success_Rate': attack_success_rate
        })
        
        print(f"    ε={eps}: {adv_acc:.4f} (ASR: {attack_success_rate:.1f}%)")

# Save FGSM results
fgsm_df = pd.DataFrame(fgsm_results)
fgsm_df.to_csv('results/fgsm_attack_results.csv', index=False)
print("\n  ✓ Saved: results/fgsm_attack_results.csv")

# ============================================================================
# STEP 4: PGD-LIKE ATTACKS
# ============================================================================
print("\n[4/5] Generating PGD-like attacks...")

iterations_list = [5, 10, 20]
pgd_results = []

for name, model in models_dict.items():
    print(f"\n  Attacking {name}...")
    
    # Get baseline accuracy
    y_pred_clean = model.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred_clean)
    
    for eps in [0.05, 0.1]:  # Focus on moderate epsilon values
        for max_iter in iterations_list:
            # Generate adversarial examples
            X_adv = generate_pgd_like_perturbations(X_test, eps, max_iter)
            
            # Evaluate on adversarial examples
            y_pred_adv = model.predict(X_adv)
            adv_acc = accuracy_score(y_test, y_pred_adv)
            
            # Calculate attack success rate
            attack_success_rate = (baseline_acc - adv_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
            
            pgd_results.append({
                'Model': name,
                'Epsilon': eps,
                'Iterations': max_iter,
                'Baseline_Accuracy': baseline_acc,
                'Adversarial_Accuracy': adv_acc,
                'Attack_Success_Rate': attack_success_rate
            })
            
            print(f"    ε={eps}, iter={max_iter}: {adv_acc:.4f} (ASR: {attack_success_rate:.1f}%)")

# Save PGD results
pgd_df = pd.DataFrame(pgd_results)
pgd_df.to_csv('results/pgd_attack_results.csv', index=False)
print("\n  ✓ Saved: results/pgd_attack_results.csv")

# ============================================================================
# STEP 5: GENERATE VISUALIZATIONS
# ============================================================================
print("\n[5/5] Generating visualizations...")

# 1. FGSM Attack Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy Degradation
for name in models_dict.keys():
    model_data = fgsm_df[fgsm_df['Model'] == name]
    ax1.plot(model_data['Epsilon'], model_data['Adversarial_Accuracy'], 
            marker='o', linewidth=2, label=name, markersize=8)

ax1.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Adversarial Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('FGSM-like Attack: Accuracy Degradation', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.3, 1.0])

# Plot 2: Attack Success Rate
for name in models_dict.keys():
    model_data = fgsm_df[fgsm_df['Model'] == name]
    ax2.plot(model_data['Epsilon'], model_data['Attack_Success_Rate'], 
            marker='s', linewidth=2, label=name, markersize=8)

ax2.set_xlabel('Epsilon (ε)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('FGSM-like Attack Success Rate', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/fgsm_attack_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/fgsm_attack_analysis.png")
plt.close()

# 2. PGD Attack Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, eps_val in enumerate([0.05, 0.1]):
    for name in models_dict.keys():
        model_data = pgd_df[(pgd_df['Model'] == name) & (pgd_df['Epsilon'] == eps_val)]
        axes[idx].plot(model_data['Iterations'], model_data['Adversarial_Accuracy'], 
                      marker='o', linewidth=2, label=name, markersize=8)
    
    axes[idx].set_xlabel('Iterations', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Adversarial Accuracy', fontsize=12, fontweight='bold')
    axes[idx].set_title(f'PGD-like Attack (ε={eps_val})', fontsize=13, fontweight='bold')
    axes[idx].legend(fontsize=10)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].set_ylim([0.3, 1.0])

plt.tight_layout()
plt.savefig('results/pgd_attack_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/pgd_attack_analysis.png")
plt.close()

# 3. Comparison: FGSM vs PGD
fig, ax = plt.subplots(figsize=(10, 6))

# Select epsilon=0.1 for comparison
fgsm_eps01 = fgsm_df[fgsm_df['Epsilon'] == 0.1]
pgd_eps01_iter20 = pgd_df[(pgd_df['Epsilon'] == 0.1) & (pgd_df['Iterations'] == 20)]

x = np.arange(len(models_dict))
width = 0.35

bars1 = ax.bar(x - width/2, fgsm_eps01['Adversarial_Accuracy'].values, 
              width, label='FGSM-like (ε=0.1)', color='#3498db')
bars2 = ax.bar(x + width/2, pgd_eps01_iter20['Adversarial_Accuracy'].values, 
              width, label='PGD-like 20 iter (ε=0.1)', color='#e74c3c')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Adversarial Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Attack Comparison: FGSM-like vs PGD-like (ε=0.1)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([m.replace(' ', '\n') for m in models_dict.keys()], fontsize=9)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/attack_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/attack_comparison.png")
plt.close()

# 4. Vulnerability Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

# Create vulnerability matrix (Attack Success Rate)
vuln_matrix = []
model_names = list(models_dict.keys())
for name in model_names:
    model_asr = fgsm_df[fgsm_df['Model'] == name]['Attack_Success_Rate'].values
    vuln_matrix.append(model_asr)

vuln_matrix = np.array(vuln_matrix)

im = ax.imshow(vuln_matrix, cmap='YlOrRd', aspect='auto')

# Set ticks
ax.set_xticks(np.arange(len(epsilons)))
ax.set_yticks(np.arange(len(model_names)))
ax.set_xticklabels([f'ε={e}' for e in epsilons])
ax.set_yticklabels(model_names)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Attack Success Rate (%)', rotation=270, labelpad=20, fontweight='bold')

# Add text annotations
for i in range(len(model_names)):
    for j in range(len(epsilons)):
        text = ax.text(j, i, f'{vuln_matrix[i, j]:.1f}%',
                      ha="center", va="center", color="black", fontsize=10)

ax.set_title('Model Vulnerability Heatmap (FGSM-like)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/vulnerability_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/vulnerability_heatmap.png")
plt.close()

print("\n" + "="*80)
print("COMPONENT 2 COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  Results:")
print("    - results/fgsm_attack_results.csv")
print("    - results/pgd_attack_results.csv")
print("\n  Visualizations:")
print("    - results/fgsm_attack_analysis.png")
print("    - results/pgd_attack_analysis.png")
print("    - results/attack_comparison.png")
print("    - results/vulnerability_heatmap.png")
print("\nKey Findings:")
max_vuln = fgsm_df.loc[fgsm_df['Attack_Success_Rate'].idxmax()]
print(f"  • Most vulnerable: {max_vuln['Model']} at ε={max_vuln['Epsilon']}")
print(f"  • Highest attack success: {max_vuln['Attack_Success_Rate']:.1f}%")
print(f"  • Average robustness drop at ε=0.1: {fgsm_df[fgsm_df['Epsilon']==0.1]['Accuracy_Drop'].mean():.3f}")
print("\nNote: Using perturbation-based attacks compatible with all model types")
print("      (tree-based models like Random Forest don't support gradient-based attacks)")
print("\nNext: Run component3_defense_mechanisms.py")
print("="*80)
