# 🍽️ Restaurant Review Sentiment Analysis (NLP)

A Natural Language Processing (NLP) project that classifies restaurant reviews as positive or negative using a Bag of Words model and a Naive Bayes classifier. The project covers the full NLP pipeline — from text cleaning and stemming to feature extraction and model evaluation.

---

## 📁 Dataset

- **File:** `Restaurant_Reviews.tsv`
- **Rows:** 1,000 reviews
- **Format:** Tab-separated (TSV), `quoting=3` to handle special characters

| Column | Description |
|---|---|
| `Review` | Raw text of the restaurant review |
| `Liked` | Target label — `1` = Positive, `0` = Negative |

---

## 🔍 Project Workflow

### 1. Text Preprocessing
Each review was cleaned and normalized through the following steps:

1. **Remove non-alphabetic characters** using regex (`re.sub`)
2. **Lowercase** all text
3. **Tokenize** by splitting on whitespace
4. **Remove stopwords** using NLTK's English stopword list — with `"not"` kept intentionally (preserves negation meaning for sentiment)
5. **Stemming** using Porter Stemmer — reduces words to their root form (e.g. *"running"* → *"run"*)

> Two stemming approaches were explored: `PorterStemmer` and `RegexpStemmer` (suffix-based: `ing`, `ed`, `ly`)

### 2. Feature Extraction — Bag of Words
- `CountVectorizer` with `max_features=1500`
- Produces a sparse matrix where each column represents a word token frequency

### 3. Train/Test Split
- 80/20 split with `random_state=0`

### 4. Model — Gaussian Naive Bayes
- Trained on the BoW feature matrix
- Evaluated using accuracy score and confusion matrix

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Accuracy Score | Overall percentage of correct predictions |
| Confusion Matrix | Breakdown of TP, TN, FP, FN |

---

## 🛠️ Tech Stack

- **Language:** Python 3
- **Libraries:** pandas, NumPy, NLTK, scikit-learn, Matplotlib

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/mahmoudibrahim55033/<repo-name>.git
   cd <repo-name>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLTK stopwords (runs automatically in the notebook, or manually):
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. Launch the notebook:
   ```bash
   jupyter notebook restaurant-review.ipynb
   ```

> Make sure `Restaurant_Reviews.tsv` is in the same directory as the notebook.

---

## 👤 Author

**mahmoudibrahim55033** — [GitHub Profile](https://github.com/mahmoudibrahim55033)
