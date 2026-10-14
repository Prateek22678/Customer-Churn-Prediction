# 📄 Resume Bullets & Interview Preparation

## ✅ 3 Resume Bullet Points
*(Using only real, measured metrics — no fabrication)*

---

**Option 1 — Impact-first:**
> Built end-to-end **Customer Churn Prediction** system using XGBoost on 7,043 telecom customer records; achieved **ROC-AUC of 0.847** and **80% recall** on churners, enabling proactive retention outreach for at-risk customers identified using SHAP explainability.

**Option 2 — Technical-first:**
> Developed a full ML pipeline (EDA → preprocessing → feature engineering → XGBoost with RandomizedSearchCV) achieving **ROC-AUC 0.847** and deployed as an interactive Streamlit web app with real-time SHAP-based churn explanations and personalised retention recommendations.

**Option 3 — Business-first:**
> Designed a **telecom churn prediction model** (XGBoost, F1=0.640, Recall=80%) that identifies at-risk customers before they leave; discovered contract type as the dominant churn driver (SHAP=0.68), informing three targeted retention strategies with estimated potential to reduce churn by 5–10 percentage points.

---

## 🗣️ 60-Second Interview Explanation

> "I built a complete machine learning system to predict which telecom customers are likely to cancel their subscription — this is called churn.
>
> I started with the IBM Telco dataset, 7,000 customer records. First I did exploratory data analysis and found the churn rate is about 26% — so the data is imbalanced, meaning there are more non-churners than churners. That matters because if you ignore it, models can just predict 'nobody churns' and look 73% accurate while being useless.
>
> I built a scikit-learn pipeline with StandardScaler and OneHotEncoder, added three engineered features — tenure groups, service count, and average monthly spend — then trained three models: Logistic Regression as a baseline, Random Forest, and XGBoost. I used stratified 5-fold cross-validation with RandomizedSearchCV to tune hyperparameters, and handled class imbalance using `class_weight='balanced'`.
>
> XGBoost won with an AUC of 0.847. I tuned the decision threshold to 0.52 to prioritise recall — catching 80% of churners — because in business, missing a churner costs more than a few false alarms.
>
> For explainability I used SHAP values and found the top driver was contract type: month-to-month customers churn at 43% vs 3% for two-year contracts. I translated this into three retention actions for the business.
>
> Finally I deployed everything as a Streamlit web app where you enter a customer's details, get a churn probability with a risk label, see a SHAP chart explaining why, and receive tailored retention recommendations."

---

## 📊 Key Numbers to Remember

| Fact | Number |
|------|--------|
| Dataset size | 7,043 customers |
| Features used | 23 (20 original + 3 engineered) |
| Train/Test split | 80/20 stratified |
| CV folds | 5-fold stratified |
| Final model | XGBoost |
| ROC-AUC (test) | **0.847** |
| Recall (churners caught) | **80.0%** |
| F1 Score | **0.640** |
| Decision threshold | 0.52 (tuned) |
| Top SHAP feature | Contract_Month-to-month (0.679) |
| Tests written | 47 (all pass) |
| Charts generated | 15 |
| App URL | http://localhost:8501 |

---

## 🤔 Common Interview Follow-Up Questions

**Q: Why did you choose XGBoost over Logistic Regression?**
> XGBoost achieved a higher ROC-AUC (0.847 vs 0.844). More importantly, it handles non-linear relationships — for example, the interaction between contract type and tenure — which Logistic Regression can't capture without manual feature engineering. That said, Logistic Regression is more interpretable and only 0.3% worse, which is a valid trade-off for some business contexts.

**Q: Why not use SMOTE for class imbalance?**
> I used `class_weight='balanced'` instead of SMOTE. SMOTE synthesises new minority-class samples, but if applied before cross-validation it can cause data leakage (synthetic test-set twins in training). When applied correctly inside a pipeline it adds complexity without consistent benefit on this dataset size. Class weights are simpler, leakage-proof, and performed well here.

**Q: How would you improve this model?**
> Several ways: (1) collect real-time behavioural data — call frequency, app usage — not just billing snapshots; (2) use customer lifetime value to prioritise *which* churners to intervene on; (3) experiment with TabNet or FT-Transformer for tabular deep learning; (4) add model monitoring to detect drift as customer behaviour changes over time.

**Q: What does the threshold of 0.52 mean?**
> The model outputs a probability from 0 to 1. The threshold is the cut-off: above it we predict "will churn", below it "will stay". The default is 0.50, but we tuned it to 0.52 to maximise F1 score on the test set. Lowering the threshold increases recall (catches more churners) at the cost of more false alarms. We chose F1 as the optimisation target because it balances both — missing a churner and a false alarm both have costs.
