# MSc CTI Robustness and Explainability Framework

**Robust and Explainable Machine Learning for Predictive Cyber Threat Intelligence**

Implementation of an integrated framework for adversarially robust and explainable cyber threat intelligence systems.

## 👥 Authors

- Muhammad Sajid Iqbal (Component 1: Baseline Models)
- Aneela Shafique (Component 2: Adversarial Attacks)
- Muhammad Arfan (Component 3: Defense Mechanisms)
- Muhammad Talha Anwar (Component 4: Explainability)

**Institution:** University of the West of Scotland  
**Programme:** MSc Information Technology  
**Supervisor:** Dr. Haider Ali

---

## 📋 Project Overview

This project addresses critical vulnerabilities in machine learning-based cyber threat intelligence systems by:

1. **Building baseline ML models** for phishing URL detection
2. **Assessing adversarial vulnerabilities** through FGSM and PGD attacks
3. **Implementing defense mechanisms** via adversarial training
4. **Providing explainability** through SHAP and LIME analysis

---

## 🎯 Research Objectives

- Develop high-accuracy baseline models for CTI threat detection
- Quantify adversarial vulnerability across different model architectures
- Implement and evaluate defense mechanisms to close robustness gaps
- Ensure model interpretability through explainable AI techniques

---

## 📊 Dataset

**Source:** Kaggle Phishing Websites Dataset  
**Samples:** 22,110 URLs  
**Features:** 30 URL-based characteristics  
**Classes:** Binary (Phishing / Legitimate)

Dataset includes structural, lexical, and content-based URL features for comprehensive phishing detection.

---

## 🛠️ Installation

### Prerequisites

- Python 3.9.7 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/MSc-CTI-Robustness-Project.git
cd MSc-CTI-Robustness-Project
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- pandas, numpy (data handling)
- scikit-learn, imbalanced-learn (machine learning)
- adversarial-robustness-toolbox (attacks/defenses)
- shap, lime (explainability)
- matplotlib, seaborn (visualization)

---

## 🚀 Usage

### Option 1: Run All Components (Recommended)

```bash
python run_all.py
```

Executes all four components sequentially (~20-30 minutes).

### Option 2: Run Individual Components

```bash
# Component 1: Train baseline models
python component1_baseline_models.py

# Component 2: Generate adversarial attacks
python component2_adversarial_attacks.py

# Component 3: Implement defenses
python component3_defense_mechanisms.py

# Component 4: Explainability analysis
python component4_explainability.py
```

---

## 📁 Project Structure

```
MSc-CTI-Project/
│
├── Dataset_Phising_Website.csv          # Input data
│
├── component1_baseline_models.py        # Baseline model training
├── component2_adversarial_attacks.py    # FGSM/PGD attacks
├── component3_defense_mechanisms.py     # Adversarial training
├── component4_explainability.py         # SHAP/LIME analysis
│
├── run_all.py                           # Master execution script
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
│
├── results/                             # Generated results
│   ├── *.csv                           # Performance metrics
│   └── *.png                           # Visualizations
│
└── models/                              # Trained models
    └── *.pkl                           # Serialized models
```

---

## 📊 Components

### Component 1: Baseline Predictive Models

**Responsibility:** Muhammad Sajid Iqbal

**Implementation:**
- Logistic Regression (linear baseline)
- Random Forest (ensemble method)
- Support Vector Machine with RBF kernel

**Outputs:**
- Confusion matrices
- ROC curves  
- Performance metrics (accuracy, precision, recall, F1)
- Cross-validation results

**Key Results:**
- Random Forest: 98.46% accuracy
- SVM: 96.29% accuracy
- Logistic Regression: 92.58% accuracy

---

### Component 2: Adversarial Attack Assessment

**Responsibility:** Aneela Shafique

**Implementation:**
- FGSM attacks (ε = 0.01, 0.05, 0.1, 0.2)
- PGD attacks (iterations: 5, 10, 20)
- Transferability analysis

**Outputs:**
- Attack success rates
- Accuracy degradation curves
- Transferability matrices

**Key Results:**
- Maximum attack success rate: 47% (FGSM ε=0.2 on RF)
- PGD more effective than FGSM
- Moderate cross-model transferability

---

### Component 3: Defense Mechanisms

**Responsibility:** Muhammad Arfan

**Implementation:**
- Adversarial training (ratios: 20/80, 30/70, 40/60)
- Isolation Forest anomaly detection
- Input sanitization

**Outputs:**
- Robustness gap reduction
- Defense effectiveness comparisons
- Anomaly detection rates

**Key Results:**
- Adversarial accuracy improved from 52% to 99.89%
- Optimal ratio: 30/70 adversarial/clean
- Isolation Forest: 73% adversarial detection rate

---

### Component 4: Explainable AI

**Responsibility:** Muhammad Talha Anwar

**Implementation:**
- SHAP (TreeExplainer for Random Forest)
- LIME (local perturbation-based)
- Consistency analysis

**Outputs:**
- Feature importance rankings
- SHAP-LIME correlation
- Baseline vs defended comparison

**Key Results:**
- SHAP-LIME correlation: ρ = 0.78-0.81
- Top features: SSL certificate, URL anchors, web traffic
- Feature importance shift after defense: ρ = 0.49

---

## 📈 Results

All results are saved in the `results/` folder:

### CSV Files:
- `baseline_results.csv` - Model performance metrics
- `fgsm_attack_results.csv` - FGSM attack data
- `pgd_attack_results.csv` - PGD attack data
- `defense_results.csv` - Defense effectiveness
- `explainability_results.csv` - SHAP-LIME consistency

### Visualizations:
- Confusion matrices for all models
- ROC curves comparison
- Attack effectiveness plots
- Defense comparison charts
- Feature importance rankings
- SHAP summary plots

---

## 🔬 Methodology Highlights

### Data Preprocessing
- 80/20 stratified train-test split
- SMOTE for class balancing (training only)
- StandardScaler normalization
- Feature extraction from URL characteristics

### Model Training
- GridSearchCV hyperparameter optimization
- 5-fold stratified cross-validation
- Random seed (42) for reproducibility

### Attack Generation
- Adversarial Robustness Toolbox (ART) library
- White-box threat model assumption
- L∞ perturbation budget constraints

### Defense Implementation
- Adversarial training with mixed datasets
- Isolation Forest (contamination = 0.1)
- Feature constraint validation

### Explainability
- SHAP TreeExplainer (exact Shapley values)
- LIME with 1000-5000 perturbations
- Spearman rank correlation for consistency

---

## 📚 Key Findings

1. **Baseline Performance:** Random Forest achieves 98.46% accuracy, outperforming SVM (96.29%) and Logistic Regression (92.58%)

2. **Adversarial Vulnerability:** Even high-performing models show significant vulnerability, with accuracy dropping to 52% under FGSM attacks

3. **Defense Effectiveness:** Adversarial training restores adversarial accuracy to 99.89% with minimal clean accuracy degradation

4. **Explainability:** SHAP and LIME demonstrate strong agreement (ρ > 0.78), validating interpretation consistency

5. **Feature Importance:** SSL certificate status, URL anchors, and web traffic emerge as critical phishing indicators

---

## 🤝 Contributing

This is an academic research project for MSc dissertation. 

For questions or collaboration inquiries, please contact the authors through the University of the West of Scotland.

---

## 📄 License

This project is submitted as part of MSc IT dissertation requirements at the University of the West of Scotland.

All code implements standard machine learning and adversarial robustness techniques using open-source libraries.

---

## 🙏 Acknowledgments

- **Supervisor:** Dr. Haider Ali (University of the West of Scotland)
- **Dataset:** Kaggle Phishing Websites Dataset contributors
- **Libraries:** scikit-learn, ART, SHAP, LIME development teams

---

## 📞 Contact

For academic inquiries related to this research:

**Programme:** MSc Information Technology  
**Institution:** University of the West of Scotland  
**Academic Year:** 2025-2026

---

## ⚙️ Technical Specifications

**Development Environment:**
- Python 3.9.7
- scikit-learn 1.0.2
- adversarial-robustness-toolbox 1.10.1
- SHAP 0.41.0
- LIME 0.2.0.1

**Hardware Requirements:**
- Minimum: 4GB RAM, Dual-core CPU
- Recommended: 8GB RAM, Quad-core CPU
- Execution time: 20-30 minutes on standard hardware

---

**Last Updated:** April 2026
