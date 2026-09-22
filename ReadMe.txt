# HealthConnect: Model Testing, Refinement & Evaluation

This repository contains the model testing and refinement pipeline for the **HealthConnect Patient No-Show Prediction** project. The objective is to evaluate candidate machine learning models against a baseline to optimize **Patient No-Show (Target = 1)** detection at the time of appointment booking.

---

## Project Overview

Building on prior data cleaning and feature engineering, this stage focuses on model validation, slice analysis, and decision threshold optimization:

1. **Candidate Model Evaluation:** Compares the baseline Logistic Regression model (Week 5) against a tuned Gradient Boosting Classifier (Week 6) using Stratified Train-Test splitting.
2. **Error & Subgroup Slice Analysis:** Evaluates false negative/positive rates across key features like lead time and patient age groups, alongside permutation feature importance and overfitting checks.
3. **Threshold Refinement:** Optimizes the decision threshold to prioritize **Recall** for practical clinic reminder workflows.
4. **Engineering Handoff & Governance:** Exports model performance metrics, configuration settings, and deployment directives for production integration.

---

## Tech Stack

* **Python 3.x**
* **Pandas & NumPy:** Data loading, feature interaction engineering, and slice manipulation.
* **Scikit-Learn:** Model fitting (`LogisticRegression`, `GradientBoostingClassifier`), evaluation metrics, stratified splits, permutation importance, and precision-recall curve analysis.
* **Matplotlib & Seaborn:** Visualization styling.

---

## Dataset & Input Requirements

The script expects a preprocessed dataset named `Cleaned_Data.csv` in the root directory. Key columns include:

* `target`: Binary target variable (1 = No-Show, 0 = Attended).
* `lead_days` / `booking_lead_days`: Days between booking and the appointment.
* `no_show_rate`, `age`, `previous_no_shows`, `distance_to_clinic_km`, `previous_appointments`: Primary predictive feature set.
* `age_group`: Categorical feature used for subgroup slice evaluation.

---

## Pipeline & Refinement Steps

### 1. Data Preparation & Feature Interaction
* Loads `Cleaned_Data.csv` and sets `target` as the label.
* Engineers interaction features (`lead_days_x_no_show_rate` and `age_x_lead_days`).
* Drops low-value/noisy features (e.g., `is_weekend`).
* Performs a 80/20 Stratified Train-Test split (`random_state=42`).

### 2. Candidate Model Testing
* **Baseline Model:** Evaluates scaled numeric features using a balanced `LogisticRegression` model.
* **Candidate Model:** Trains a tuned `GradientBoostingClassifier` (`n_estimators=250`, `learning_rate=0.05`, `max_depth=3`, `subsample=0.85`).
* Generates classification performance metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

### 3. Error Analysis, Slice Evaluation & Overfitting Checks
* Categorizes test predictions into **False Negatives**, **False Positives**, and **Correct** predictions.
* Computes mean lead days across error categories and recall across `age_group` subgroups.
* Measures feature importance using ROC-AUC Permutation Importance on the test set.
* Calculates Train vs. Test ROC-AUC delta to assess model overfitting.

### 4. Threshold Refinement & ML Engineering Handoff
* Computes the precision-recall curve to select an optimal decision threshold (~0.42).
* Shifts the decision threshold to boost No-Show Recall (improving recall to ~0.798).
* Outputs a summary metrics comparison table (`week7_refinement_results.csv`) and prints the ML Engineering handoff configuration.

### 5. Deployment Suitability & Risk Assessment
* Prints clinical operational directives: model is suitable for automated triage and reminder triggers, but not for hard operational cancellations.
* Highlights key risks (signal ceiling, temporal drift, cold-start handling) and required next steps for temporal validation and API monitoring.

---

## Outputs

When executed, the notebook produces:
* **Terminal Summary Logs:** Comparisons of model performance metrics, error counts, top permutation features, and handoff payloads.
* **`week7_refinement_results.csv`:** Pre- and post-refinement performance evaluation table saved to disk.
