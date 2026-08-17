# AI-Based Sentiment & Emotion Detection System

A dual-task NLP project that performs:

- **Emotion Detection** (6 classes): `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`
- **Sentiment Analysis** (3 classes): `positive`, `negative`, `neutral`

Built using **TF-IDF + Logistic Regression** and deployed with **Streamlit**.

---

## 🚀 Live Demo

👉 [Try the app on Streamlit](https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/)

---

## 📌 Problem Statement

Understanding human text requires identifying both:
1. The **emotion** being expressed, and  
2. The overall **sentiment polarity**.

This project demonstrates a practical end-to-end NLP pipeline that handles both tasks in one workflow.

---

## ✨ Key Features

- Dual-task text classification (emotion + sentiment)
- Text preprocessing pipeline (regex cleaning, stopword removal, normalization)
- TF-IDF feature extraction (unigrams + bigrams)
- Logistic Regression models for both tasks
- Validation using accuracy, classification report, and confusion matrix
- Interactive/manual inference testing
- Streamlit deployment for real-time predictions

---

## 🧠 Learning Outcomes

This project covers:

- NLP preprocessing fundamentals
- Multi-class classification setup
- Feature engineering with TF-IDF
- Performance evaluation and class-wise error analysis
- Converting notebook experiments into reusable scripts
- Basic ML model deployment with Streamlit

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** `pandas`, `nltk`, `scikit-learn`, `matplotlib`
- **ML Methods:** TF-IDF, Logistic Regression
- **Deployment:** Streamlit

---

## 📂 Repository Structure

```text
.
├── NLP_Project_Sentiment_Analysis.ipynb      # Notebook: training + evaluation workflow
├── nlp_project_sentiment_analysis.py         # Script: reproducible pipeline
└── README.md
```

---

## ⚙️ Workflow

1. Load training / validation / test datasets  
2. Map numeric emotion labels to class names  
3. Derive sentiment labels from emotion labels  
4. Clean and normalize text  
5. Vectorize text with TF-IDF  
6. Train models:
   - Emotion classifier (class-weighted Logistic Regression)
   - Sentiment classifier (Logistic Regression)
7. Evaluate on validation data  
8. Perform manual input testing  
9. Visualize confusion matrices and class distributions

---

## ▶️ Run Locally

### 1) Clone repository
```bash
git clone https://github.com/drishtichaudhary/ai-sentiment-emotion-analysis.git
cd ai-sentiment-emotion-analysis
```

### 2) Install dependencies
```bash
pip install pandas nltk scikit-learn matplotlib streamlit
```

### 3) Run training/evaluation script
```bash
python nlp_project_sentiment_analysis.py
```

### 4) (Optional) Run Streamlit app
```bash
streamlit run app.py
```

> **Note:** Update dataset paths in your script (e.g., `training.csv.xls`, `validation.csv.xls`, `test.csv.xls`) to match your local directory.

---

## 📊 Example Output

For a given input sentence, the system predicts:

- **Emotion Class**
- **Sentiment Class**

It also generates evaluation artifacts such as:

- Accuracy scores
- Classification reports
- Confusion matrices

---

## ⚠️ Current Limitations

- Uses classical ML models (no contextual transformer embeddings)
- Performance depends on preprocessing and dataset quality
- Sentiment mapping via emotion labels may introduce label-noise assumptions

---

## 🔮 Future Improvements

- Upgrade to transformer-based models (BERT/RoBERTa/DistilBERT)
- Add hyperparameter tuning + k-fold cross-validation
- Improve class imbalance handling (resampling / focal loss alternatives)
- Add confidence scores and probability distributions
- Enhance Streamlit UI with analytics dashboard and prediction history
- Package project with `requirements.txt` and modular code structure

---

## 👩‍💻 Author

**Drishti Chaudhary**  
GitHub: [@drishtichaudhary](https://github.com/drishtichaudhary)
Linkedin: [Drishti Chaudhary](https://www.linkedin.com/in/drishti-chaudhary/)
---

## 📄 License

This project is open-source.  
If you plan to share or reuse it publicly, consider adding an MIT License file.
