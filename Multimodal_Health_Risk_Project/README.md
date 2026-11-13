# Multimodal Health Risk Prediction Using Explainable AI

Academic prototype for disease-specific health-risk analysis using four modalities:
Laboratory Reports, Medical History, Vital Signs, and Medical Imaging findings.

Diseases:
- Diabetes
- Heart Disease
- Bronchitis
- Breast Cancer

## Important dataset note
The supplied workbook is synthetic. The application is for engineering-project development and demonstration, not clinical diagnosis or validated medical use.

## Main workflow
Excel dataset -> preprocessing -> multimodal feature fusion -> disease model -> risk level -> SHAP/LIME explanation -> prevention -> next possible disease -> next-disease explanation/prevention.

## Run in VS Code / terminal

### 1. Create environment
Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install packages
```bash
pip install -r requirements.txt
```

### 3. Train models
```bash
python src/train_models.py
```

### 4. Start Flask
```bash
python app.py
```

Open http://127.0.0.1:5000

## Files
- `src/preprocessing.py` - dataset loading and preprocessing
- `src/train_models.py` - model comparison/training
- `src/explainability.py` - SHAP/LIME
- `src/predictor.py` - prediction and recommendations
- `app.py` - Flask application
- `templates/` - web pages
- `static/` - CSS and JavaScript
