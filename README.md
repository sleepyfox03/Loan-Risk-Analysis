# Loan Default Risk Prediction — with Fairness & Explainability

Predicting loan default risk from applicant data, with an explicit focus on **which model tradeoff actually serves the business decision**, and **whether the model is fair across sex and age** — not just chasing accuracy.

## Problem

A lender needs to decide which applicants are likely to default. Missing a real defaulter (false negative) costs the full loan principal; flagging a good applicant as risky (false positive) costs a lost opportunity — but at a much smaller cost. That asymmetry should shape which model and which threshold get used, not just which one scores highest on accuracy.

## Dataset

[German Credit Data](https://github.com/selva86/datasets/blob/master/GermanCredit.csv) (UCI Machine Learning Repository, via selva86/datasets) — 1,000 applicants, 20 raw features, 30% default rate, no missing values.

## Approach

1. **EDA** — class balance, distributions, default rate by category, correlation structure. Surfaced a genuinely counter-intuitive finding (applicants with no tracked checking account default *less*, not more) and reported it honestly rather than smoothing it over.
2. **Feature engineering** — repayment-burden proxy, risk-history flags, loan-term buckets. Sensitive attributes (`sex`, `age_group`) were extracted separately and deliberately excluded from the model's input features.
3. **Model comparison** — 3 model families (Logistic Regression, Random Forest, XGBoost) x 2 imbalance-handling strategies (class weighting vs SMOTE) = 6 models, evaluated on precision, recall, F1, and ROC-AUC on a held-out test set.
4. **Explainability** — SHAP values on the best model to identify and sanity-check the top drivers of predicted risk.
5. **Fairness audit** — Fairlearn `MetricFrame`, demographic parity difference, and equalized odds difference, computed by sex and by age group, even though neither was a model input.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (class-weighted)** | 0.565 | **0.800** | 0.662 | **0.803** |
| XGBoost (SMOTE) | 0.643 | 0.600 | 0.621 | 0.799 |
| Logistic Regression (SMOTE) | 0.547 | 0.783 | 0.644 | 0.793 |
| Random Forest (class-weighted) | 0.667 | 0.400 | 0.500 | 0.788 |
| Random Forest (SMOTE) | 0.652 | 0.500 | 0.566 | 0.787 |
| XGBoost (class-weighted) | 0.578 | 0.617 | 0.597 | 0.783 |

**Best model:** class-weighted Logistic Regression — highest ROC-AUC and highest recall (catches 80% of real defaulters), which matters most for this cost asymmetry. Its precision (0.565) is a real limitation: roughly 44% of applicants flagged as high-risk are false alarms, which is why I'd recommend a human-review layer for flagged applicants rather than auto-rejection.

**Top risk drivers (SHAP):** checking-account status, installment burden relative to loan, loan duration, loan amount, and credit history — mostly matching domain intuition, with one flagged anomaly (see notebook) worth validating with a domain expert before full trust.

**Fairness:**
- **By sex:** effectively fair. Demographic parity difference of 0.002; near-identical selection rates and recall.
- **By age:** a real disparity. Applicants under 25 are flagged as high-risk 53.8% of the time vs 39.8% for 25+, a demographic parity difference of 0.141 — even though age was never given to the model. Full discussion and mitigation recommendations are in the notebook.

## Repo structure

```
loan-default-project/
├── data/
│   └── GermanCredit.csv              # raw dataset
├── notebooks/
│   └── loan_default_risk_analysis.ipynb   # full analysis, fully executed
├── figures/                          # all charts as standalone PNGs
├── README.md
└── requirements.txt
```

## Running it yourself

```bash
pip install -r requirements.txt
jupyter notebook notebooks/loan_default_risk_analysis.ipynb
```

## What I'd do with more time

- Try Fairlearn's `ExponentiatedGradient` to train a fairness-constrained version and measure the accuracy/fairness tradeoff directly
- Validate the counter-intuitive `no_checking_account` finding against domain expertise rather than trusting it at face value
- Test a decision threshold tuned specifically to the false-negative/false-positive cost ratio, instead of the default 0.5 cutoff
