import os
import pickle
import re

import fitz
import streamlit as st


DISCLAIMER = (
    "This system predicts whether a news article is likely fake or likely real based on "
    "text patterns learned from a labeled dataset. It does not verify facts, sources, "
    "or real-world truth. Human fact-checking is still required."
)


def clean_text(text):
    """Clean and normalize article text for prediction."""
    if text is None:
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_pickle_file(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def load_model_files():
    model_path = "model/fake_news_model.pkl"
    vectorizer_path = "model/vectorizer.pkl"

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        return None, None

    return load_pickle_file(model_path), load_pickle_file(vectorizer_path)


def extract_text_from_pdf(uploaded_file):
    text_parts = []

    try:
        pdf_bytes = uploaded_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            for page in document:
                page_text = page.get_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as error:
        st.error(f"Could not read the PDF file. Details: {error}")
        return ""

    return "\n".join(text_parts).strip()


def extract_pdf_text(uploaded_file):
    return extract_text_from_pdf(uploaded_file)


def show_result_card(result_title, explanation, result_type):
    st.markdown(
        f"""
        <div class="result-card {result_type}">
            <h3>{result_title}</h3>
            <p>{explanation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_confidence(fake_confidence, real_confidence):
    highest_confidence = max(fake_confidence, real_confidence)
    fake_column, real_column = st.columns(2)

    fake_column.metric("Fake Confidence", f"{fake_confidence * 100:.2f}%")
    real_column.metric("Real Confidence", f"{real_confidence * 100:.2f}%")

    st.progress(float(highest_confidence))
    st.caption(f"Highest confidence: {highest_confidence * 100:.2f}%")


def add_custom_css():
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2.5rem;
            padding-bottom: 2.5rem;
        }

        .app-header {
            border-bottom: 1px solid #e5e7eb;
            margin-bottom: 1.5rem;
            padding-bottom: 1.25rem;
        }

        .app-header h1 {
            color: #111827;
            font-size: 2rem;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }

        .app-header p {
            color: #4b5563;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }

        .app-header span {
            color: #6b7280;
            font-size: 3rem;
        }

        .section-title {
            color: #111827;
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }

        .result-card {
            border-radius: 8px;
            margin: 1rem 0;
            padding: 1rem 1.1rem;
        }

        .result-card h3 {
            font-size: 1.25rem;
            margin: 0 0 0.35rem 0;
        }

        .result-card p {
            margin: 0;
        }

        .result-fake {
            background: #fef2f2;
            border: 1px solid #fecaca;
            color: #7f1d1d;
        }

        .result-real {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            color: #14532d;
        }

        .result-uncertain {
            background: #fffbeb;
            border: 1px solid #fde68a;
            color: #78350f;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(
        page_title="NLP Fake News Detector",
        page_icon="📰",
        layout="centered",
    )
    add_custom_css()

    st.markdown(
        """
        <div class="app-header">
            <h1>NLP-Based Fake News Detection Prototype</h1>
            <p>A machine learning prototype that classifies news text as likely fake or likely real using TF-IDF and Logistic Regression.</p>
            
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, vectorizer = load_model_files()
    if model is None or vectorizer is None:
        st.error("Model files are missing. Please run: python train_model.py")
        st.stop()

    st.markdown('<div class="section-title">Analyze News Article</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Choose input method",
        ["Paste Text", "Upload PDF"],
        horizontal=True,
    )

    article_text = ""

    if input_mode == "Paste Text":
        article_text = st.text_area("Paste news article text", height=250)
    else:
        uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

        if uploaded_pdf is not None:
            article_text = extract_text_from_pdf(uploaded_pdf)

            if article_text:
                st.success("PDF text extracted successfully.")
                with st.expander("Preview Extracted PDF Text"):
                    st.write(article_text[:3000])
            else:
                st.warning(
                    "No readable text was found. This may be a scanned or image-based "
                    "PDF and may require OCR."
                )

    if st.button("Analyze Article", type="primary"):
        if not article_text.strip():
            st.warning("Please paste text or upload a readable PDF first.")
        else:
            cleaned_text = clean_text(article_text)

            if not cleaned_text:
                st.warning("Please paste text or upload a readable PDF first.")
            else:
                text_features = vectorizer.transform([cleaned_text])
                prediction = model.predict(text_features)[0]
                probabilities = model.predict_proba(text_features)[0]

                class_indexes = {label: index for index, label in enumerate(model.classes_)}
                fake_confidence = probabilities[class_indexes.get(0, 0)]
                real_confidence = probabilities[class_indexes.get(1, 1)]
                highest_confidence = max(fake_confidence, real_confidence)

                if highest_confidence < 0.60:
                    show_result_card(
                        "Uncertain / Needs Verification",
                        "The model confidence is low, so this article should be manually checked.",
                        "result-uncertain",
                    )
                elif prediction == 0:
                    show_result_card(
                        "Likely Fake News",
                        "The article text contains patterns that are closer to fake-news examples learned during training.",
                        "result-fake",
                    )
                else:
                    show_result_card(
                        "Likely Real News",
                        "The article text contains patterns that are closer to real-news examples learned during training.",
                        "result-real",
                    )

                show_confidence(fake_confidence, real_confidence)

    with st.expander("How this prototype works"):
        st.write("1. The user enters article text or uploads a PDF.")
        st.write("2. The system extracts readable PDF text if a file is uploaded.")
        st.write("3. The text is cleaned and normalized.")
        st.write("4. TF-IDF converts the text into numerical features.")
        st.write("5. Logistic Regression classifies the article as likely fake or likely real.")
        st.write("6. The system displays the prediction and confidence score.")

    with st.expander("Limitations and Disclaimer"):
        st.write(DISCLAIMER)
        st.write("- The system may misclassify satire or sarcasm.")
        st.write("- The system depends on the quality and size of the training dataset.")
        st.write(
            "- Synthetic/sample datasets are useful for prototype testing but not enough "
            "for real-world accuracy."
        )
        st.write("- PDF extraction only works for readable text-based PDFs.")
        st.write("- Scanned or image-based PDFs may require OCR.")


if __name__ == "__main__":
    main()
