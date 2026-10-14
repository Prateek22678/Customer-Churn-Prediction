# 🚀 Deployment Guide — Churn Prediction App

## Option A: Streamlit Community Cloud (Free, Recommended)

### Prerequisites
- GitHub account (free at github.com)
- Streamlit account (free at share.streamlit.io, sign in with GitHub)

### Step 1 — Install Git
Download from: https://git-scm.com/download/win
During install: accept all defaults, make sure "Add Git to PATH" is checked.
Verify: open a new terminal and run `git --version`

### Step 2 — Push to GitHub
```bash
# In your project folder:
git init
git add .

# The .gitignore already excludes data/, venv/, and .pkl files.
# For deployment you MUST commit the model file — temporarily allow it:
git add -f models/xgboost_pipeline.pkl

git commit -m "Initial commit: Customer Churn Prediction ML project"

# Create a new repo on github.com (name: churn-prediction, public)
# Then connect and push:
git remote add origin https://github.com/Prateek22678/Customer-Churn-Prediction.git
git branch -M main
git push -u origin main
```

> ⚠️ **Model file size:** The XGBoost pipeline .pkl is ~5MB. GitHub allows files up to 100MB,
> so this is fine. For larger models, use Git LFS.

### Step 3 — Deploy on Streamlit Cloud
1. Go to **https://share.streamlit.io**
2. Click **"New app"**
3. Select your `churn-prediction` GitHub repo
4. Set **Main file path**: `app/streamlit_app.py`
5. Set **Python version**: 3.11 (or closest available)
6. Click **"Deploy!"**

Streamlit will:
- Install all packages from `requirements.txt` automatically
- Build your app
- Give you a public URL like: `https://YOUR-USERNAME-churn-prediction.streamlit.app`

### Step 4 — Fix for Deployment (Important!)
On Streamlit Cloud, the app can't find the model using a relative path.
Add this to the top of `app/streamlit_app.py` (already included):
```python
ROOT_DIR = Path(__file__).parent.parent
model_path = ROOT_DIR / "models" / "xgboost_pipeline.pkl"
```
This works both locally and on Streamlit Cloud. ✅

---

## Option B: Hugging Face Spaces (Alternative)

### Step 1 — Create a Space
1. Go to **https://huggingface.co/spaces**
2. Click **"New Space"**
3. Name: `churn-prediction`
4. SDK: **Streamlit**
5. Visibility: Public

### Step 2 — Upload Files
Upload these files/folders:
- `app/streamlit_app.py`
- `src/` (entire folder)
- `models/xgboost_pipeline.pkl`
- `requirements.txt`

In your Space's `README.md`, set the entry point:
```yaml
---
title: Customer Churn Predictor
sdk: streamlit
app_file: app/streamlit_app.py
---
```

### Step 3 — Set the App File
In Space settings, set: **App file** → `app/streamlit_app.py`

Your app will be live at:
`https://huggingface.co/spaces/YOUR_USERNAME/churn-prediction`

---

## Option C: Run Locally (Always Works)

```bash
# Activate virtual environment
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Run the app
streamlit run app/streamlit_app.py

# App opens at: http://localhost:8501
```

---

## 📸 Getting Your App Screenshot (For README)

1. Run the app locally: `streamlit run app/streamlit_app.py`
2. Open `http://localhost:8501` in your browser
3. Fill in a sample high-risk customer:
   - Tenure: 2 months
   - Contract: Month-to-month
   - Internet: Fiber optic
   - Monthly charges: $85
   - Payment: Electronic check
4. Click **"Predict Churn Risk"**
5. Take a screenshot (Win+Shift+S on Windows)
6. Save as `reports/figures/app_screenshot.png`
7. Add to README: `![App Screenshot](reports/figures/app_screenshot.png)`

---

## 📋 Manual Checklist Before Submitting to GitHub

- [ ] Dataset downloaded and working (`data/raw/*.csv`)
- [ ] All 47 tests pass: `pytest tests/ -v`
- [ ] Models saved: `models/*.pkl`
- [ ] All 15 figures saved in `reports/figures/`
- [ ] App runs: `streamlit run app/streamlit_app.py`
- [ ] App screenshot taken and added to README
- [ ] Git installed and configured
- [ ] GitHub repo created (public)
- [ ] Code pushed to GitHub
- [ ] Streamlit Cloud deployment created
- [ ] Public URL working and shared on LinkedIn/Resume
