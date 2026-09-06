# AI-Based Sentiment & Emotion Detection System

A dual-task NLP classification system that predicts both the **emotion** expressed in text and the corresponding **sentiment polarity**. The project uses a classical supervised learning pipeline built around **TF-IDF feature engineering** and **Logistic Regression**, and exposes the workflow through a **Streamlit** interface.

## Live Demo

Try the deployed app on Streamlit: https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/

## Overview

This repository implements an applied NLP pipeline for two related but distinct text-classification tasks:

1. **Emotion classification** over 6 classes: `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`
2. **Sentiment classification** over 3 classes: `positive`, `negative`, `neutral`

The project is intentionally scoped as a **classical machine-learning baseline** rather than a deep-learning or transformer-based system. It is useful for studying the effect of preprocessing, sparse lexical features, class imbalance handling, and supervised multiclass classification in a reproducible NLP workflow.

## Problem Formulation

**Input:** natural-language text

**Outputs:**
- an emotion label
- a sentiment label

The implementation maps numeric emotion IDs to named emotion categories and then derives sentiment labels from those emotion categories using a rule-based mapping:
- `joy` and `love` → `positive`
- `sadness`, `anger`, and `fear` → `negative`
- `surprise` → `neutral`

This sentiment formulation is a methodological choice in the current implementation and should be interpreted accordingly.

## Dataset

The notebook/script reads three dataset files:
- `training.csv.xls`
- `validation.csv.xls`
- `test.csv.xls`

The source files are loaded with pandas and expected to contain at least the columns `text` and `label`. The label mapping used in the code is:
- `0 -> sadness`
- `1 -> joy`
- `2 -> love`
- `3 -> anger`
- `4 -> fear`
- `5 -> surprise`

The repository does not include dataset statistics or class-distribution summaries in the source files, so no counts or percentages are claimed here.

## NLP Preprocessing

The implementation applies the following text preprocessing steps:
- convert text to lowercase
- remove non-word characters with a regex-based cleaner while preserving `!` and `?`
- tokenize by whitespace
- remove English stopwords using NLTK
- reconstruct cleaned text for downstream vectorization

The cleaned datasets are also written to CSV files during execution:
- `train_cleaned.csv`
- `val_cleaned.csv`
- `test_cleaned.csv`

## Feature Engineering

Text is represented using **TF-IDF** via `TfidfVectorizer` with:
- `max_features=5000`
- `ngram_range=(1, 2)`

This creates a sparse lexical representation that captures both unigram and bigram signals, which is a strong baseline for short-text classification tasks where word choice and local phrase patterns can be informative.

## Model Architecture

The repository trains two separate supervised classifiers on the same TF-IDF feature space:

### Emotion classifier
- Model: `LogisticRegression(max_iter=1000, class_weight='balanced')`
- Purpose: predict one of the 6 emotion classes
- Note: class weighting is used here to mitigate class imbalance in the emotion task

### Sentiment classifier
- Model: `LogisticRegression(max_iter=1000)`
- Purpose: predict one of the 3 sentiment classes

Both models are classical linear classifiers trained on sparse TF-IDF features. This makes the system lightweight, interpretable, and well suited to a baseline comparison against more complex NLP approaches.

## Experimental Workflow

The implemented workflow is:

1. Load training, validation, and test datasets
2. Map numeric labels to emotion names
3. Derive sentiment labels from emotion names
4. Clean and normalize text
5. Build TF-IDF features
6. Train the emotion classifier
7. Train the sentiment classifier
8. Evaluate on validation data
9. Run manual single-text inference
10. Plot confusion matrices for error inspection

## Evaluation

The code computes the following evaluation outputs on the validation split:
- accuracy
- classification report
- confusion matrix

The repository does not store the numeric evaluation results in the README or source artifacts in a way that can be safely summarized here, so no metric values are fabricated.

The notebook also includes confusion-matrix visualization for both tasks, which supports a basic form of class-wise error analysis.

## Deployment

The project is deployed as a Streamlit application for interactive inference.

When a user enters text, the workflow is:
1. clean the text with the same preprocessing function
2. transform the text with the fitted TF-IDF vectorizer
3. generate an emotion prediction
4. generate a sentiment prediction
5. display the results in the UI

## System Workflow

```text
User Text
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
Emotion Classifier + Sentiment Classifier
↓
Predicted Emotion + Sentiment
↓
Streamlit Interface
```

## Technical Design Decisions

- **TF-IDF** was chosen as a practical sparse representation for lexical text features.
- **Logistic Regression** is a strong classical baseline for multiclass text classification.
- **Separate classifiers** are used because emotion and sentiment are related but not identical prediction targets.
- **Class weighting** is applied to the emotion model to help address imbalance across emotion classes.
- **Streamlit** provides a simple interface for interactive testing and deployment.

## Research & Engineering Relevance

This is an **applied ML/NLP project**, not an academic research study. Its value for research-oriented review lies in the end-to-end experimentation workflow it demonstrates:
- supervised text classification
- sparse feature engineering
- multiclass modeling
- class imbalance handling
- validation and error analysis
- reproducible preprocessing and inference
- lightweight model deployment

## Limitations

- TF-IDF does not model contextual semantics as effectively as transformer embeddings.
- Performance depends on preprocessing choices and dataset quality.
- Sentiment labels derived from emotion categories introduce a rule-based assumption and possible label noise.
- Generalization may be limited outside the training distribution.
- No transformer-based contextual encoder is used in the current implementation.

## Future Improvements

Possible next steps include:
- compare TF-IDF + Logistic Regression against BERT, DistilBERT, or RoBERTa
- perform hyperparameter optimization
- add stratified k-fold cross-validation
- explore alternative class-imbalance strategies
- calibrate prediction confidence
- add probability visualizations in the UI
- conduct more systematic error analysis
- test robustness on noisy or out-of-domain text
- modularize preprocessing, training, and inference code
- add a reproducible dependency file such as `requirements.txt`

## Repository Structure

```text
.
├── NLP_Project_Sentiment_Analysis.ipynb   # Notebook with data loading, preprocessing, training, evaluation, and manual testing
├── nlp_project_sentiment_analysis.py      # Script version of the main NLP pipeline
└── README.md
```

## Running Locally

### 1) Clone the repository

```bash
git clone https://github.com/drishtichaudhary/ai-sentiment-emotion-analysis.git
cd ai-sentiment-emotion-analysis
```

### 2) Install dependencies

```bash
pip install pandas nltk scikit-learn matplotlib streamlit
```

### 3) Provide the dataset files

Make sure the expected dataset files are available at the paths used in the notebook/script, or update those paths before running:
- `training.csv.xls`
- `validation.csv.xls`
- `test.csv.xls`

### 4) Run the training/evaluation pipeline

```bash
python nlp_project_sentiment_analysis.py
```

### 5) Run the Streamlit app

If your local repository includes a Streamlit entry point, run it with the appropriate file name. The current repository contents shown here do not verify the presence of `app.py`.

## Technologies / Skills Demonstrated

### NLP / Machine Learning
- NLP preprocessing
- TF-IDF feature engineering
- Logistic Regression
- multiclass classification
- validation and classification reports
- confusion-matrix-based error inspection
- class imbalance handling

### Python / Data Science
- Python
- pandas
- NLTK
- scikit-learn
- matplotlib

### Deployment
- Streamlit

### Engineering
- reproducible ML workflow
- data preprocessing
- model evaluation
- interactive inference

## Outcome

The project delivers a lightweight dual-task NLP system that predicts emotion and sentiment from text using classical supervised learning, with both notebook-based experimentation and a Streamlit-based application layer.

## Screenshots / Demonstration

_No screenshots are included in the repository at the moment._

## Author

Drishti Chaudhary

GitHub: https://github.com/drishtichaudhary

LinkedIn: https://www.linkedin.com/in/drishti-chaudhary/

## License

This repository does not currently include a LICENSE file in the source shown here. If you plan to reuse or distribute the project, consider adding an explicit license such as MIT.
