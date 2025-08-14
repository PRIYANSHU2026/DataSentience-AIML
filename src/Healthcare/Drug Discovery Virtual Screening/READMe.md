# Drug Discovery Virtual Screening – Compound Activity Classifier

## 📌 Overview
This project builds a machine learning classification model to predict whether a chemical compound is **active** or **inactive** against a target protein based on molecular and binding features.
[!ui](assets/image.png)
The goal is to assist researchers in prioritizing compounds for **drug discovery pipelines** by automating the **virtual screening** step.

---

## 📊 Dataset
The dataset contains various molecular descriptors and protein-binding features, including:

- `binding_site_size`
- `molecular_weight`
- `polar_surface_area`
- `h_bond_donors`
- `rotatable_bonds`
- `protein_pi`
- `hydrophobicity`
- `h_bond_acceptors`
- `mw_ratio`
- `logp_pi_interaction`
- `protein_length`
- `logp`
- `compound_clogp`
- `binding_affinity`
- `active` (target variable – 1 for active, 0 for inactive)

---

## 🛠 Model Pipeline
The classification pipeline includes:
1. **Preprocessing** – handling missing values, scaling numerical features
2. **Model Training** – RandomForestClassifier for robust classification
3. **Evaluation** – accuracy, precision, recall, and F1-score
4. **Serialization** – saving trained model for later predictions

---

## 🚀 How to Run

### 1️⃣ Install Dependencies
```bash
pip install pandas scikit-learn joblib

python train.py
python predict.py
