# Bank Customer Churn Prediction Using Machine Learning

## Overview
This project predicts whether a bank customer is likely to leave the bank (customer churn) using machine learning. It compares **Logistic Regression** and **Random Forest** across three datasets and evaluates them using multiple classification metrics.

This is an experimental analysis and decision-support project, not a production-ready banking system.

## Objectives
- Explore customer churn patterns in different datasets.
- Clean and preprocess customer data.
- Train and evaluate Logistic Regression and Random Forest.
- Compare Accuracy, Precision, Recall, F1-score, and ROC-AUC.
- Explore how changing the classification threshold affects churn detection.

## Datasets
Each dataset is evaluated separately. They are not merged because their columns and customer populations differ.

| Dataset | File | Rows | Churn rate | Target |
|---|---|---:|---:|---|
| Churn Modelling | `Churn_Modelling.csv` | 10,000 | 20.37% | `Exited` |
| Playground Series S4E1 | `train.csv` | 165,034 | 21.16% | `Exited` |
| BankChurners | `BankChurners.csv` | 10,127 | 16.07% | `Attrition_Flag` |

The datasets were sourced from Kaggle. Obtain them from their original dataset pages and follow the applicable licensing and usage terms.

## Models Used

### Logistic Regression
Used as a baseline binary-classification model. It estimates the probability that a customer will churn and converts that probability into a class prediction using a threshold.

### Random Forest
Combines predictions from multiple decision trees. This project uses 200 trees and `class_weight="balanced"` to give additional weight to the minority churn class.

## Workflow
1. Load each dataset.
2. Inspect dimensions, missing values, duplicates, and target distribution.
3. Remove identifier columns that are not useful for prediction.
4. For BankChurners, remove the `Naive_Bayes_Classifier_...` columns to reduce target-leakage risk.
5. Convert the target to binary labels: `1` = churn and `0` = no churn.
6. Split into training and testing sets using an 80/20 split, `random_state=42`, and stratification.
7. Preprocess numerical and categorical features:
   - Numerical features: median imputation.
   - Categorical features: most-frequent imputation and one-hot encoding.
   - Logistic Regression: numerical features are scaled.
   - Random Forest: numerical features are not scaled.
8. Train both models.
9. Evaluate predictions on the held-out test set.
10. Test a Random Forest probability threshold of 0.35 and compare it with 0.50.

Preprocessing is fitted on training data through pipelines to reduce data leakage.

## Evaluation Metrics
- **Accuracy:** Proportion of all predictions that are correct.
- **Precision:** Of customers predicted to churn, the proportion who actually churned.
- **Recall:** Of customers who actually churned, the proportion correctly identified.
- **F1-score:** Harmonic mean of Precision and Recall.
- **ROC-AUC:** Measures how well the model distinguishes churners from non-churners across different classification thresholds.

Accuracy alone can be misleading when churn is the minority class, so Recall, F1-score, and ROC-AUC are also considered.

## Results
Recorded results at a classification threshold of 0.50:

| Dataset | Model | Accuracy | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| D1 — Churn Modelling | Logistic Regression | 0.808 | 0.187 | 0.284 | 0.775 |
| D1 — Churn Modelling | Random Forest | 0.862 | 0.437 | 0.562 | 0.855 |
| D2 — Playground S4E1 | Logistic Regression | 0.834 | 0.381 | 0.492 | 0.815 |
| D2 — Playground S4E1 | Random Forest | 0.860 | 0.523 | 0.612 | 0.874 |
| D3 — BankChurners | Logistic Regression | 0.901 | 0.542 | 0.637 | 0.917 |
| D3 — BankChurners | Random Forest | 0.951 | 0.742 | 0.830 | 0.985 |

These values are from the recorded notebook run and should be reproduced before being treated as final performance results.

### Threshold Experiment
The Random Forest threshold was reduced from 0.50 to 0.35. Recall increased while Precision decreased: the model caught more churners but also flagged more customers who did not churn.

| Dataset | Recall at 0.50 | Recall at 0.35 | Precision at 0.50 | Precision at 0.35 | F1 at 0.50 | F1 at 0.35 |
|---|---:|---:|---:|---:|---:|---:|
| D1 — Churn Modelling | 0.437 | 0.587 | 0.788 | 0.651 | 0.562 | 0.618 |
| D2 — Playground S4E1 | 0.523 | 0.649 | 0.738 | 0.638 | 0.612 | 0.643 |
| D3 — BankChurners | 0.742 | 0.862 | 0.941 | 0.870 | 0.830 | 0.866 |

## Main Findings
- Random Forest outperformed Logistic Regression on Recall, F1-score, and ROC-AUC across all three recorded datasets.
- Accuracy alone does not show how many likely churners were missed.
- Lowering the probability threshold increased Recall but reduced Precision.
- Results differ across datasets; their scores should be interpreted in the context of their different customer populations and features.
- The unusually high ROC-AUC reported for BankChurners should be independently validated before drawing strong conclusions.

## Limitations
- One stratified train/test split was used per dataset.
- Extensive hyperparameter tuning and cross-validation were not performed.
- Feature importance shows model associations, not causation.
- The datasets represent different customer populations and are not directly interchangeable.
- The 0.35 threshold is an experiment, not a universally optimal business cutoff.
- Fairness, operational impact, data drift, and production monitoring have not been fully evaluated.

## Future Improvements
- Use stratified cross-validation and repeat experiments with different splits.
- Tune model hyperparameters.
- Compare with gradient-boosting models.
- Select a threshold based on the business costs of missed churners and false alarms.
- Check fairness and model behaviour across relevant customer groups.
- Validate data provenance and possible leakage, especially for the high BankChurners ROC-AUC.
- Test on recent, representative data before considering deployment.

## Requirements
A typical environment includes:
- Python 3.10+
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook or JupyterLab

Install common dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Exact requirements may vary depending on the notebook imports.

## How to Run
1. Download or clone the project repository.
2. Place the CSV files in the paths expected by the notebook, or update the file paths.
3. Install the dependencies.
4. Open the notebook in Jupyter Notebook, JupyterLab, or a compatible environment.
5. Run the notebook cells in order.
6. Review the metrics and threshold comparison.

**Note:** Dataset filenames and paths must match those used in the notebook.

## Suggested Project Structure

```text
bank-customer-churn-prediction/
├── README.md
├── notebook.ipynb
├── data/
│   ├── Churn_Modelling.csv
│   ├── train.csv
│   └── BankChurners.csv
└── results/
```

This is a suggested layout; adjust it to match your actual files. Do not publish private customer data or large datasets without permission.

## Conclusion
This project demonstrates a machine-learning workflow for customer churn prediction, compares a baseline model with an ensemble model, and shows how a probability threshold affects the balance between Recall and Precision. Further validation is required before using the model for real banking decisions.

## Author
**Arpan Munshi**  
B.Tech — Computer Science and Engineering  
Symbiosis Institute of Technology, Nagpur
