# NLP-Based Fake News Detection Prototype Using TF-IDF and Logistic Regression

## Project Purpose

This project is a beginner-friendly academic prototype for detecting whether a news article is likely fake or likely real. It uses simple Natural Language Processing techniques with TF-IDF features and a Logistic Regression classifier.

The goal is to keep the system easy to run, easy to understand, and suitable for a class presentation.

## Technologies Used

- Python
- pandas
- scikit-learn
- Streamlit
- PyMuPDF

## Dataset Requirement

Place the dataset files inside the `data` folder:

- `data/Fake.csv`
- `data/True.csv`

Both CSV files must contain these columns:

- `title`
- `text`

Rows from `Fake.csv` are labeled as fake news. Rows from `True.csv` are labeled as real news.

## Folder Structure

```text
fake-news-detector/
|-- data/
|   |-- Fake.csv
|   `-- True.csv
|-- model/
|   |-- fake_news_model.pkl
|   |-- vectorizer.pkl
|   `-- evaluation.json
|-- train_model.py
|-- app.py
|-- requirements.txt
`-- README.md
```

The files inside the `model` folder are generated automatically after training. Do not create them manually.

## Install Requirements

Run this command in the project folder:

```bash
python -m pip install -r requirements.txt
```

## Train the Model

After placing `Fake.csv` and `True.csv` inside the `data` folder, run:

```bash
python train_model.py
```

This creates:

- `model/fake_news_model.pkl`
- `model/vectorizer.pkl`
- `model/evaluation.json`

The training script also prints the accuracy, classification report, and confusion matrix.

## Run the App

Start the Streamlit app with:

```bash
python -m streamlit run app.py
```

The app lets you paste article text or upload a PDF file for analysis.

## How PDF Upload Works

PDF upload uses PyMuPDF to extract readable text from the uploaded file. If the PDF contains selectable text, the app can analyze it.

If the PDF is scanned or image-based, the app may not find readable text. In that case, OCR would be required, but OCR is not included in this prototype.

## System Limitations

- The system does not perform real fact-checking.
- The prediction depends on the training dataset.
- The model may be wrong when articles are very short, unclear, or very different from the training data.
- The PDF feature only extracts readable text and does not read scanned images.
- This project does not use deep learning, BERT, transformers, APIs, databases, or login systems.

## Disclaimer

"This system predicts whether a news article is likely fake or likely real based on text patterns learned from a labeled dataset. It does not verify facts, sources, or real-world truth. Human fact-checking is still required."
