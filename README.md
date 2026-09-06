# AI-Based Sentiment & Emotion Detection System

An applied dual-task NLP/ML system that predicts **6 emotion classes** and **3 sentiment classes** from text. It uses **text preprocessing, TF-IDF feature engineering, and Logistic Regression**, with an interactive **Streamlit** deployment for inference.

---

## Live Demo

Streamlit deployment:

https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/

---

## Overview

This repository implements a classical supervised NLP pipeline for two related classification tasks:

- **Emotion classification**: predict one of 6 emotion labels
- **Sentiment classification**: predict one of 3 sentiment labels

The project is intentionally scoped as a **classical machine-learning baseline** rather than a transformer-based system. Its technical value lies in the end-to-end workflow: label design, preprocessing, sparse feature extraction, multiclass classification, validation, manual inference, and deployment.

---

## Problem Formulation

**Input:** natural-language text

**Outputs:**
- emotion label
- sentiment label

### Emotion classification

Predict one of the following 6 emotion classes:

- `sadness`
- `joy`
- `love`
- `anger`
- `fear`
- `surprise`

### Sentiment classification

Predict one of the following 3 sentiment classes:

- `positive`
- `negative`
- `neutral`

---

## Label Design

The repository implements the following numeric-to-emotion mapping:

| Numeric label | Emotion |
|---|---|
| 0 | sadness |
| 1 | joy |
| 2 | love |
| 3 | anger |
| 4 | fear |
| 5 | surprise |

Sentiment labels are derived from emotion labels using the following mapping:

- `joy` + `love` → `positive`
- `sadness` + `anger` + `fear` → `negative`
- `surprise` → `neutral`

This is an explicit modeling choice in the implementation. Therefore, the sentiment task is constructed from the emotion labels rather than being independently annotated in the source data.

---

## Dataset

The notebook/script loads three dataset files:

- `training.csv.xls`
- `validation.csv.xls`
- `test.csv.xls`

The code expects at least the following columns:

- `text`
- `label`

The data is used as follows:

- **Training set**: model fitting
- **Validation set**: evaluation
- **Test set**: preprocessing and cleaned-output generation

The repository does not currently provide stable dataset statistics or class-distribution summaries, so no such values are claimed here.

---

## NLP Preprocessing

The implementation applies a lightweight text-normalization pipeline:

- lowercase conversion
- regex-based cleaning
- preservation of `!` and `?`
- whitespace tokenization
- NLTK English stopword removal
- reconstruction of cleaned text

The preprocessing stage also writes cleaned dataset outputs during execution:

- `train_cleaned.csv`
- `val_cleaned.csv`
- `test_cleaned.csv`

This preprocessing provides a consistent representation for the downstream classical ML models.

---

## Feature Engineering

The text is represented using **TF-IDF** through `TfidfVectorizer` with:

- `max_features=5000`
- `ngram_range=(1, 2)`

This produces a sparse lexical representation using:

- **unigrams**
- **bigrams**

TF-IDF provides an efficient classical baseline for capturing discriminative lexical and short phrase-level patterns in text.

---

## Model Design

The repository trains two separate Logistic Regression classifiers over the same TF-IDF feature space.

### Emotion Classifier

```text
LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)
````

The balanced class weighting is used to help address class imbalance across the emotion categories.

### Sentiment Classifier

```text
LogisticRegression(
    max_iter=1000
)
```

The two classifiers share the same feature representation but learn separate target spaces for emotion and sentiment.

---

## Experimental Design

The implemented workflow is:

1. Load training, validation, and test datasets
2. Map numeric labels to emotion names
3. Derive sentiment labels from emotion categories
4. Clean and normalize text
5. Build TF-IDF features
6. Train the emotion classifier
7. Train the sentiment classifier
8. Validate both models on held-out data
9. Generate classification reports
10. Visualize confusion matrices
11. Run manual single-text inference

This makes the project a compact applied ML experiment rather than solely a deployment-focused application.

---

## Evaluation

The repository evaluates the models using:

* **accuracy**
* **classification reports**
* **confusion matrices**

The notebook/script prints accuracy and class-wise classification reports for the validation data, while confusion matrices are used for error inspection.

No specific metric values are reproduced in this README because the repository does not currently store them in a stable, verifiable form suitable for citation.

---

## Deployment

The trained NLP workflow is exposed through a **Streamlit** interface for interactive inference.

The prediction flow is:

1. User enters text
2. The preprocessing function cleans the input
3. TF-IDF transforms the cleaned text
4. The emotion classifier predicts an emotion
5. The sentiment classifier predicts a sentiment
6. The predictions are displayed through the Streamlit interface

The deployed application is available through the live demo link above.

> The Streamlit deployment source is not currently included in this repository. The README therefore documents the deployed application without claiming that a local Streamlit entry file is present.

---

## System Workflow

```text
User Text
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
 ┌─────────────────────────────┐
 │                             │
 ↓                             ↓
Emotion Classifier      Sentiment Classifier
 │                             │
 ↓                             ↓
Emotion Label            Sentiment Label
 │                             │
 └──────────────┬──────────────┘
                ↓
        Streamlit Interface
```

---

## Research & Engineering Relevance

This is an **applied ML/NLP project**, not an academic research paper.

For research-oriented evaluation, the project demonstrates an end-to-end experimental workflow involving:

* dual-task problem formulation
* explicit label engineering
* NLP preprocessing
* sparse feature representation
* supervised multiclass classification
* class imbalance handling
* validation-based evaluation
* confusion-matrix-based error analysis
* manual inference testing
* deployment of an NLP model

The project also establishes a classical ML baseline that can be systematically compared against contextual transformer-based approaches in future experiments.

---

## Technical Design Decisions

### TF-IDF

TF-IDF was selected as a computationally efficient sparse representation capable of capturing discriminative lexical patterns and short phrases through unigram and bigram features.

### Logistic Regression

Logistic Regression provides an efficient and interpretable linear baseline for multiclass classification over sparse text features.

### Separate Classifiers

Emotion and sentiment are related but distinct prediction targets, so separate classifiers are trained over the shared TF-IDF representation.

### Class Weighting

`class_weight="balanced"` is applied to the emotion classifier to help compensate for differences in class frequency.

### Streamlit

Streamlit provides a lightweight interface for interactive model inference and deployment.

---

## Limitations

The current implementation has several methodological limitations:

* TF-IDF does not capture contextual semantics as effectively as transformer-based representations.
* The sentiment labels inherit assumptions from the emotion-to-sentiment mapping.
* The sentiment task is therefore not equivalent to training on an independently annotated sentiment dataset.
* Generalization depends on the distribution and quality of the underlying dataset.
* No transformer-based contextual model is implemented.
* No formal k-fold cross-validation is included.
* No external benchmark comparison is included.
* The current repository does not include the Streamlit deployment source.

These limitations also define several natural directions for further experimentation.

---

## Future Research Directions

The following are proposed future directions and are **not implemented in the current repository**:

* Compare TF-IDF + Logistic Regression against BERT, DistilBERT, or RoBERTa
* Investigate independently annotated sentiment datasets
* Perform hyperparameter optimization
* Add stratified k-fold cross-validation
* Conduct ablation studies on preprocessing and n-gram choices
* Perform systematic class-wise error analysis
* Evaluate robustness on noisy and out-of-domain text
* Calibrate model confidence and probability estimates
* Compare alternative feature representations
* Investigate whether jointly modeling emotion and sentiment improves predictive performance

---

## Repository Structure

```text
.
├── NLP_Project_Sentiment_Analysis.ipynb   # Notebook with data loading, preprocessing,
│                                          # model training, evaluation, and manual testing
├── nlp_project_sentiment_analysis.py      # Script version of the NLP pipeline
└── README.md
```

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/drishtichaudhary/ai-sentiment-emotion-analysis.git
cd ai-sentiment-emotion-analysis
```

### 2. Install dependencies

```bash
pip install pandas nltk scikit-learn matplotlib streamlit
```

Download the NLTK English stopword corpus:

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### 3. Provide the dataset files

Make sure the dataset files expected by the notebook/script are available at the paths used in the code, or update those paths before running:

* `training.csv.xls`
* `validation.csv.xls`
* `test.csv.xls`

### 4. Run the training/evaluation script

```bash
python nlp_project_sentiment_analysis.py
```

### 5. Streamlit deployment

The deployed application is available at:

[https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/](https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/)

The repository does not currently contain the original Streamlit deployment source, so a local Streamlit command is not specified here.

---

## Technologies / Skills

### NLP / Machine Learning

* NLP preprocessing
* TF-IDF feature engineering
* Logistic Regression
* multiclass classification
* class imbalance handling
* validation-based evaluation
* classification reports
* confusion-matrix-based error analysis

### Python / Data Science

* Python
* pandas
* NLTK
* scikit-learn
* matplotlib

### Deployment

* Streamlit

### Engineering

* data preprocessing
* supervised ML workflow
* model evaluation
* manual inference
* interactive model deployment

---

## Demonstration

Live Streamlit application:

[https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/](https://ai-based-sentiment-emotion-detection-system-spt6xi6i5tdpzr6qub.streamlit.app/)

---

## Author

**Drishti Chaudhary**

* **GitHub:** [https://github.com/drishtichaudhary](https://github.com/drishtichaudhary)
* **LinkedIn:** [https://www.linkedin.com/in/drishtichaudhary-047855206](https://www.linkedin.com/in/drishtichaudhary-047855206)
* **Portfolio:** [https://drishtichaudhary.github.io/Portfolio-site/](https://drishtichaudhary.github.io/Portfolio-site/)

---

## License

This repository does not currently include a LICENSE file.

If the project is intended for public reuse or distribution, an explicit open-source license should be added.
