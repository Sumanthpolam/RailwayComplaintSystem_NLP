# Railway Complaint Classification System — GitHub Ready

This project includes two ways to use the system:

1. **Static frontend for GitHub Pages** — `index.html`, `style.css`, and `script.js`. This works on GitHub Pages because the complaint classification demo runs in the browser using keyword-based rules.
2. **Python ML app** — `app.py` is the original Streamlit application and loads the included trained model files in `models/`.

> Important: GitHub Pages only hosts static files. It cannot run Python, Streamlit, or the `.pkl` model. The GitHub Pages frontend is a working demo classifier using JavaScript rules. For real inference with the trained Python model, deploy the Streamlit app separately on Streamlit Community Cloud.

## Project structure

```text
RailwayComplaintSystem_NLP/
├── index.html                  # GitHub Pages frontend
├── style.css                   # Frontend styles
├── script.js                   # Browser-based demo classifier
├── app.py                      # Original Streamlit ML application
├── train_pipeline.py           # Model training pipeline
├── requirements.txt
├── models/
│   ├── complaint_model.pkl
│   └── tfidf_vectorizer.pkl
└── data/
    └── After_NLP_Preprocessing.csv
```

The virtual environment (`.venv`) is intentionally excluded. Do not upload a virtual environment to GitHub.

## Publish the frontend on GitHub Pages

1. Extract this ZIP.
2. Copy the **contents of this folder** into the root of your GitHub repository `RailwayComplaintSystem_NLP`.
3. Commit and push the files:
   ```bash
   git add .
   git commit -m "Add responsive frontend for GitHub Pages"
   git push origin main
   ```
4. Open the repository on GitHub and go to **Settings → Pages**.
5. Under **Build and deployment**, select **Deploy from a branch**.
6. Select branch `main` and folder `/ (root)`, then click **Save**.
7. Wait for the Pages deployment to complete. The website will be available at:
   `https://sumanthpolam.github.io/RailwayComplaintSystem_NLP/`

Make sure `index.html`, `style.css`, and `script.js` are in the repository root, not inside an extra nested project folder.

## Run the Python ML application locally

Use Python 3.10 or 3.11 for broad compatibility with the installed model dependencies.

Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints, usually `http://localhost:8501`.

## Deploy the Python app separately

For model-based inference, deploy the repository as a Streamlit app at Streamlit Community Cloud and set the main file path to `app.py`. Keep `models/complaint_model.pkl` and `models/tfidf_vectorizer.pkl` available in the deployed repository.

## Limitations

- The GitHub Pages frontend uses JavaScript keyword rules, not the Python-trained model.
- The generated reference is a demo identifier. It does not file a complaint with Indian Railways.
- The included `train_pipeline.py` expects `data/Messy_DATA.csv`; that raw file was not present in the uploaded ZIP. The currently trained model files are included, so retraining requires restoring the raw dataset.
- Never commit passenger-sensitive information, credentials, or secrets.
