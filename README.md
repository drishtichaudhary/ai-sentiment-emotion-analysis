# AI-Based Sentiment & Emotion Detection System

A dual-task NLP project that predicts:

- **Emotion** (6 classes): sadness, joy, love, anger, fear, surprise  
- **Sentiment** (3 classes): positive, negative, neutral  

This project uses **TF-IDF vectorization** + **Logistic Regression** and is deployed with **Streamlit**.

## 🚀 Live Demo

👉 https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/

## 📌 Project Overview

The system takes user text as input and performs:

1. **Emotion Detection**  
   Classifies text into one of six emotion categories.

2. **Sentiment Analysis**  
   Predicts whether the text expresses a positive, negative, or neutral sentiment.

This experiment demonstrates a practical NLP workflow from preprocessing to model training, evaluation, and deployment.

## 🧠 Topics Covered in This Experiment

- Natural Language Processing (NLP) pipeline
- Text cleaning and preprocessing
- Stopword removal with NLTK
- Label mapping for multi-class emotion classification
- Rule-based mapping from emotion → sentiment
- Feature extraction using **TF-IDF** (unigram + bigram)
- Multi-class classification using **Logistic Regression**
- Model evaluation with:
  - Accuracy score
  - Classification report
  - Confusion matrix
- Basic exploratory data analysis (class distributions)
- Interactive inference (manual text testing)
- Streamlit deployment

## 🛠️ Tech Stack

- **Language:** Python  
- **Libraries:** pandas, nltk, scikit-learn, matplotlib  
- **ML Techniques:** TF-IDF, Logistic Regression  
- **Deployment:** Streamlit  

## 📂 Repository Structure

- `NLP_Project_Sentiment_Analysis.ipynb` — notebook version of training + evaluation workflow  
- `nlp_project_sentiment_analysis.py` — script version of the same workflow  

## ⚙️ Workflow

1. Load training/validation/test datasets  
2. Map numeric labels to emotion names  
3. Create sentiment labels from emotions  
4. Clean text (lowercasing, regex cleanup, stopword removal)  
5. Convert text to TF-IDF vectors  
6. Train:
   - Emotion model (Logistic Regression, class-weighted)
   - Sentiment model (Logistic Regression)
7. Evaluate performance on validation set  
8. Run manual testing on custom input text  
9. Visualize confusion matrices and class distributions  

## ▶️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/drishtichaudhary/ai-sentiment-emotion-analysis.git
   cd ai-sentiment-emotion-analysis
   ```

2. Install dependencies:
   ```bash
   pip install pandas nltk scikit-learn matplotlib
   ```

3. Run script:
   ```bash
   python nlp_project_sentiment_analysis.py
   ```

> Note: Update dataset file paths in the script (`training.csv.xls`, `validation.csv.xls`, `test.csv.xls`) based on your local environment.

## 📊 Output

For any input sentence, the model returns:

- Predicted **Emotion**
- Predicted **Sentiment**

Along with evaluation artifacts such as reports and confusion matrices.

## 🔮 Future Improvements

- Try transformer-based models (e.g., BERT/RoBERTa)
- Hyperparameter tuning and cross-validation
- Better handling of imbalanced classes
- Add confidence scores and probability output
- Build a richer Streamlit UI with charts and logs

## 👩‍💻 Author

**Drishti Chaudhary**  
GitHub: https://github.com/drishtichaudhary
