# COVID-19 Vaccine Tweet Sentiment Analysis

A sentiment analysis project that classifies tweets about COVID-19 vaccines as **Positive**, **Negative**, or **Neutral**, using NLP preprocessing, TextBlob-based polarity scoring, and machine learning classifiers.

## Overview

This project analyzes public sentiment toward COVID-19 vaccines based on tweet text. The pipeline cleans raw tweets, derives sentiment labels using `TextBlob` polarity scores, visualizes patterns through word clouds and distribution plots, and trains classifiers (Logistic Regression and Linear SVC) to predict sentiment from text.

## Dataset

The notebook expects a CSV file named `vaccination_tweets.csv` in the project directory, containing tweet metadata. Only the `text` column is used for analysis; other columns (user info, dates, hashtags, retweet counts, etc.) are dropped during preprocessing.

> **Note:** The dataset is not included in this repository. You'll need to supply your own `vaccination_tweets.csv` (e.g. from a public Kaggle dataset on COVID-19 vaccine tweets) and place it in the project root before running the notebook.

## Project Workflow

1. **Data Cleaning**
   - Drop irrelevant columns, keep only tweet text
   - Lowercase text, remove URLs, mentions, hashtags, and punctuation
   - Tokenize and remove stopwords (NLTK)
   - Remove duplicate tweets

2. **Sentiment Labeling**
   - Compute polarity scores with `TextBlob`
   - Map polarity to `Positive`, `Negative`, or `Neutral` labels

3. **Exploratory Data Analysis**
   - Sentiment distribution (count plot and pie chart)
   - Word clouds for positive, negative, and neutral tweets

4. **Feature Extraction**
   - `CountVectorizer` with unigrams and bigrams (`ngram_range=(1,2)`)

5. **Modeling**
   - Train/test split (80/20)
   - **Logistic Regression** (baseline + hyperparameter tuning via `GridSearchCV`)
   - **Linear SVC** (baseline + hyperparameter tuning via `GridSearchCV`)
   - Evaluation via accuracy, confusion matrix, and classification report

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression (default) | 84.38% |
| Logistic Regression (tuned, `C=10`) | 84.70% |
| Linear SVC (default) | 85.61% |
| Linear SVC (tuned, `C=10`) | 85.57% |

Linear SVC slightly outperformed Logistic Regression. Across both models, **Neutral** and **Positive** classes were predicted with high precision/recall, while the **Negative** class was harder to classify correctly, likely due to class imbalance (fewer negative tweets in the dataset).

## Tech Stack

- **Python 3**
- `pandas`, `numpy` — data handling
- `nltk`, `textblob` — text preprocessing & sentiment polarity
- `scikit-learn` — feature extraction (`CountVectorizer`), models (`LogisticRegression`, `LinearSVC`), `GridSearchCV`, evaluation metrics
- `matplotlib`, `seaborn`, `wordcloud` — visualization

## Installation & Setup

### Prerequisites

Make sure Python 3.x is installed on your system.

Verify installation:

```bash
python3 --version
```

### Clone the Repository

```bash
git clone <repository-url>
cd <repository-name>
```

### Create a Virtual Environment

```bash
python3 -m venv venv
```

Activate the virtual environment:

**macOS/Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Install Jupyter Lab

```bash
pip install jupyterlab
```

### Register Virtual Environment as a Jupyter Kernel

```bash
pip install ipykernel

python -m ipykernel install --user \
  --name=sentiment-analysis \
  --display-name="Python (Sentiment Analysis)"
```

### macOS SSL Certificate Fix (If NLTK Downloads Fail)

Some macOS installations may encounter SSL certificate verification errors while downloading NLTK resources.

Run:

```bash
open "/Applications/Python 3.12/Install Certificates.command"
```

Replace `3.12` with your installed Python version if different.

### Launch Jupyter Lab

```bash
jupyter lab
```

Jupyter Lab will open automatically in your browser at:

```text
http://localhost:8888
```

### Select the Correct Kernel

In Jupyter Lab:

```text
Kernel → Change Kernel → Python (Sentiment Analysis)
```

This ensures the notebook uses the virtual environment and all installed dependencies.

### Run the Notebook

Open:

```text
sentiment_analysis.ipynb
```

and execute the cells sequentially to perform data preprocessing, sentiment analysis, visualization, model training, and evaluation.

> **Note:** Make sure `vaccination_tweets.csv` is placed in the project root before running the notebook.

## Project Structure

```
.
├── sentiment_analysis.ipynb   # Main analysis notebook
├── Data
│   └── vaccination_tweets.csv # Dataset
├── requirements.txt           # Python dependencies
└── README.md
```

## Future Improvements

- Address class imbalance for the Negative sentiment class (e.g. oversampling, class weights)
- Experiment with TF-IDF or word embeddings instead of raw count vectors
- Try additional models (Naive Bayes, Random Forest, transformer-based models like BERT)
- Apply proper stemming/lemmatization (the current stemming step is a no-op)

## License

This project is open source and available under the [MIT License](LICENSE).

---

Made with ❤️ by **Somilemid04**

⭐ If you found this project helpful, consider giving it a star — it's appreciated!
