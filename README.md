## Titanic ML - Flask App

### Quick start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) Train the model explicitly:

```bash
python -m model.train
```

4. Run the app:

```bash
python app.py
```

The app will auto-train on first run if artifacts are missing.

### Notes
- Random Forest with preprocessing pipeline. Separate OneHotEncoder instances per categorical feature (`sex`, `embarked`).
- Uses the seaborn `titanic` dataset for training.
- UI built with Bootstrap 5.
