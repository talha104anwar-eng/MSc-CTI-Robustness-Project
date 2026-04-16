"""
Component 3: Defense Mechanisms Implementation
MSc IT Dissertation - CTI Robustness Project

This component implements and evaluates defense mechanisms:
- Adversarial Training (with different mixing ratios)
- Isolation Forest for anomaly detection
- Robustness evaluation

Authors: Muhammad Sajid Iqbal, Aneela Shafique, Muhammad Arfan, Muhammad Talha Anwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("COMPONENT 3: DEFENSE MECHANISMS")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA AND MODELS
# ============================================================================
print("\n[1/4] Loading data and models...")

try:
    # Load test data
    test_df = pd.read_csv('results/test_data.csv')
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Load training data (we'll recreate from original dataset)
    df = pd.read_csv('Dataset Phising Website.csv')
    if 'index' in df.columns:
        df = df.drop('index', axis=1)
    
    X = df.drop('Result', axis=1).values
    y = (df['Result'] == 1).astype(int).values
    
    # Split to get training data
    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # Load scaler
    scaler = joblib.load('models/scaler.pkl')
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  ✓ Loaded training data: {len(X_train)} samples")
    print(f"  ✓ Loaded test data: {len(X_test)} samples")
    
    # Load baseline Random Forest (we'll focus on this model)
    rf_baseline = joblib.load('models/random_forest.pkl')
    print("  ✓ Loaded baseline Random Forest model")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("  Please run component1_baseline_models.py first!")
    exit(1)

# ============================================================================
# HELPER FUNCTIONS FOR ADVERSARIAL GENERATION
# ============================================================================

def generate_fgsm_like_perturbations(X, epsilon):
    """Generate FGSM-like perturbations using random noise"""
    noise = np.random.normal(0, 1, X.shape)
    noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
    X_adv = X + epsilon * noise
    return X_adv

def generate_pgd_like_perturbations(X, epsilon, iterations=10):
    """Generate PGD-like perturbations using iterative random noise"""
    X_adv = X.copy()
    step_size = epsilon / iterations
    
    for _ in range(iterations):
        noise = np.random.normal(0, 1, X.shape)
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)
        X_adv = X_adv + step_size * noise
        # Project back to epsilon ball
        perturbation = X_adv - X
        perturbation_norm = np.linalg.norm(perturbation, axis=1, keepdims=True)
        perturbation_norm = np.maximum(perturbation_norm, 1e-10)
        perturbation = perturbation * np.minimum(epsilon / perturbation_norm, 1)
        X_adv = X + perturbation
    
    return X_adv

# ============================================================================
# STEP 2: ADVERSARIAL TRAINING
# ============================================================================
print("\n[2/4] Implementing adversarial training defenses...")

mixing_ratios = ['20/80', '30/70', '40/60']
defense_results = []

# Get baseline performance
y_pred_clean = rf_baseline.predict(X_test_scaled)
baseline_clean_acc = accuracy_score(y_test, y_pred_clean)

# Generate adversarial examples for training and testing
print("\n  Generating adversarial training examples...")
X_train_adv = generate_fgsm_like_perturbations(X_train_scaled, epsilon=0.1)
print("  ✓ Generated adversarial training samples")

# Test attack effectiveness on baseline
print("\n  Testing attack effectiveness on baseline model...")
X_test_adv = generate_fgsm_like_perturbations(X_test_scaled, epsilon=0.1)
y_pred_adv_baseline = rf_baseline.predict(X_test_adv)
baseline_adv_acc = accuracy_score(y_test, y_pred_adv_baseline)
print(f"    Baseline clean accuracy: {baseline_clean_acc:.4f}")
print(f"    Baseline adversarial accuracy: {baseline_adv_acc:.4f}")
print(f"    Robustness gap: {baseline_clean_acc - baseline_adv_acc:.4f}")

# Train defended models with different mixing ratios
for ratio_name in mixing_ratios:
    print(f"\n  Training with {ratio_name} adversarial/clean ratio...")
    
    # Parse ratio
    adv_ratio = int(ratio_name.split('/')[0]) / 100
    clean_ratio = int(ratio_name.split('/')[1]) / 100
    
    # Calculate sample sizes
    n_adv = int(len(X_train_scaled) * adv_ratio)
    n_clean = int(len(X_train_scaled) * clean_ratio)
    
    # Sample from adversarial and clean data
    adv_indices = np.random.choice(len(X_train_adv), n_adv, replace=False)
    clean_indices = np.random.choice(len(X_train_scaled), n_clean, replace=False)
    
    # Combine datasets
    X_train_mixed = np.vstack([
        X_train_adv[adv_indices],
        X_train_scaled[clean_indices]
    ])
    y_train_mixed = np.concatenate([
        y_train[adv_indices],
        y_train[clean_indices]
    ])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_train_mixed))
    X_train_mixed = X_train_mixed[shuffle_idx]
    y_train_mixed = y_train_mixed[shuffle_idx]
    
    # Train defended model
    rf_defended = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE, 
        max_depth=20
    )
    rf_defended.fit(X_train_mixed, y_train_mixed)
    
    # Evaluate on clean test set
    y_pred_clean_defended = rf_defended.predict(X_test_scaled)
    clean_acc_defended = accuracy_score(y_test, y_pred_clean_defended)
    
    # Evaluate on adversarial test set
    y_pred_adv_defended = rf_defended.predict(X_test_adv)
    adv_acc_defended = accuracy_score(y_test, y_pred_adv_defended)
    
    # Calculate improvements
    robustness_gain = adv_acc_defended - baseline_adv_acc
    robustness_gap = clean_acc_defended - adv_acc_defended
    
    defense_results.append({
        'Ratio': ratio_name,
        'Clean_Accuracy_Before': baseline_clean_acc,
        'Clean_Accuracy_After': clean_acc_defended,
        'Adversarial_Accuracy_Before': baseline_adv_acc,
        'Adversarial_Accuracy_After': adv_acc_defended,
        'Robustness_Gain': robustness_gain,
        'Robustness_Gap': robustness_gap
    })
    
    print(f"    Clean accuracy: {clean_acc_defended:.4f}")
    print(f"    Adversarial accuracy: {adv_acc_defended:.4f}")
    print(f"    Robustness gain: +{robustness_gain:.4f}")
    
    # Save defended model
    model_name = f'models/random_forest_defended_{ratio_name.replace("/", "_")}.pkl'
    joblib.dump(rf_defended, model_name)
    print(f"    ✓ Saved: {model_name}")

# Save defense results
defense_df = pd.DataFrame(defense_results)
defense_df.to_csv('results/defense_results.csv', index=False)
print("\n  ✓ Saved: results/defense_results.csv")

# ============================================================================
# STEP 3: ISOLATION FOREST DETECTION
# ============================================================================
print("\n[3/4] Testing Isolation Forest anomaly detection...")

# Train Isolation Forest on clean training data
iso_forest = IsolationForest(
    contamination=0.1,
    random_state=RANDOM_STATE
)
iso_forest.fit(X_train_scaled)

# Detect anomalies in test data
anomaly_scores_clean = iso_forest.predict(X_test_scaled)
anomaly_scores_adv = iso_forest.predict(X_test_adv)

# Calculate detection rates
clean_flagged = np.sum(anomaly_scores_clean == -1)
adv_flagged = np.sum(anomaly_scores_adv == -1)

clean_flagged_rate = clean_flagged / len(X_test_scaled) * 100
adv_flagged_rate = adv_flagged / len(X_test_adv) * 100

print(f"  Clean samples flagged: {clean_flagged}/{len(X_test_scaled)} ({clean_flagged_rate:.1f}%)")
print(f"  Adversarial samples flagged: {adv_flagged}/{len(X_test_adv)} ({adv_flagged_rate:.1f}%)")
print(f"  Detection improvement: {adv_flagged_rate - clean_flagged_rate:.1f}%")

# Save Isolation Forest
joblib.dump(iso_forest, 'models/isolation_forest.pkl')
print("  ✓ Saved: models/isolation_forest.pkl")

# ============================================================================
# STEP 4: GENERATE VISUALIZATIONS
# ============================================================================
print("\n[4/4] Generating visualizations...")

# 1. Robustness Gap Reduction
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(mixing_ratios))
width = 0.35

bars1 = ax.bar(x - width/2, defense_df['Robustness_Gap'].values, 
              width, label='Robustness Gap (After Defense)', color='#e74c3c')
bars2 = ax.bar(x + width/2, [baseline_clean_acc - baseline_adv_acc] * len(mixing_ratios), 
              width, label='Robustness Gap (Baseline)', color='#95a5a6')

ax.set_xlabel('Adversarial Training Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Robustness Gap (Clean Acc - Adv Acc)', fontsize=12, fontweight='bold')
ax.set_title('Defense Effectiveness: Robustness Gap Reduction', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mixing_ratios)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
               f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results/defense_effectiveness.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/defense_effectiveness.png")
plt.close()

# 2. Before/After Comparison
fig, ax = plt.subplots(figsize=(12, 6))

# Get best performing ratio (30/70 typically)
best_ratio_idx = 1  # 30/70
best_ratio = defense_df.iloc[best_ratio_idx]

categories = ['Clean\nAccuracy', 'Adversarial\nAccuracy', 'Robustness\nGap']
before_values = [
    baseline_clean_acc,
    baseline_adv_acc,
    baseline_clean_acc - baseline_adv_acc
]
after_values = [
    best_ratio['Clean_Accuracy_After'],
    best_ratio['Adversarial_Accuracy_After'],
    best_ratio['Robustness_Gap']
]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, before_values, width, label='Before Defense', color='#3498db')
bars2 = ax.bar(x + width/2, after_values, width, label='After Defense (30/70)', color='#2ecc71')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title(f'Defense Impact: Before vs After Adversarial Training', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim([0, 1.1])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results/defense_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/defense_comparison.png")
plt.close()

# 3. Isolation Forest Detection
fig, ax = plt.subplots(figsize=(10, 6))

detection_data = [
    ('Clean Samples', clean_flagged_rate),
    ('Adversarial Samples', adv_flagged_rate)
]

bars = ax.bar([x[0] for x in detection_data], [x[1] for x in detection_data],
             color=['#3498db', '#e74c3c'])

ax.set_ylabel('Flagged as Anomaly (%)', fontsize=12, fontweight='bold')
ax.set_title('Isolation Forest: Anomaly Detection on Clean vs Adversarial', 
            fontsize=14, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3)
ax.set_ylim([0, 100])

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
           f'{height:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('results/isolation_forest_detection.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/isolation_forest_detection.png")
plt.close()

# 4. Robustness Gain Chart
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(mixing_ratios))
robustness_gains = defense_df['Robustness_Gain'].values * 100  # Convert to percentage

bars = ax.bar(x, robustness_gains, color='#2ecc71', alpha=0.7, edgecolor='darkgreen', linewidth=2)

ax.set_xlabel('Adversarial Training Ratio', fontsize=12, fontweight='bold')
ax.set_ylabel('Robustness Gain (%)', fontsize=12, fontweight='bold')
ax.set_title('Adversarial Training: Robustness Improvement by Ratio', 
            fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(mixing_ratios)
ax.grid(True, axis='y', alpha=0.3)

# Add value labels
for bar, gain in zip(bars, robustness_gains):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
           f'+{gain:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/robustness_gain_chart.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/robustness_gain_chart.png")
plt.close()

print("\n" + "="*80)
print("COMPONENT 3 COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  Models:")
print("    - models/random_forest_defended_20_80.pkl")
print("    - models/random_forest_defended_30_70.pkl")
print("    - models/random_forest_defended_40_60.pkl")
print("    - models/isolation_forest.pkl")
print("\n  Results:")
print("    - results/defense_results.csv")
print("\n  Visualizations:")
print("    - results/defense_effectiveness.png")
print("    - results/defense_comparison.png")
print("    - results/isolation_forest_detection.png")
print("    - results/robustness_gain_chart.png")
print("\nKey Findings:")
best_defense = defense_df.loc[defense_df['Robustness_Gain'].idxmax()]
print(f"  • Best defense ratio: {best_defense['Ratio']}")
print(f"  • Adversarial accuracy improvement: +{best_defense['Robustness_Gain']:.3f}")
print(f"  • Clean accuracy maintained: {best_defense['Clean_Accuracy_After']:.3f}")
print(f"  • Isolation Forest detection rate on adversarial: {adv_flagged_rate:.1f}%")
print("\nNote: Using perturbation-based attacks compatible with Random Forest")
print("\nNext: Run component4_explainability.py")
print("="*80)
