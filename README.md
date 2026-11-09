# SMS Spam Detection

A machine-learning pipeline that classifies SMS messages as **spam** or **ham** (legitimate).
Built with classic NLP + scikit-learn classifiers, developed in Google Colab.

## Dataset

`spam.csv` — the SMS Spam Collection dataset (5,572 messages, 2 columns: label and text).
After removing 403 exact duplicates the working set is **5,169 messages**.
The classes are imbalanced: ~87% ham vs ~13% spam.

## Pipeline

1. **Loading & cleaning** — read the CSV, drop the three near-empty junk columns
   (`Unnamed: 2/3/4`), rename `v1`/`v2` to `Target`/`Text`, label-encode the target,
   drop duplicate rows.
2. **Exploratory analysis** — class balance pie chart, message length/word/sentence-count
   distributions per class, correlation heatmap, spam vs ham word clouds and word frequencies.
3. **Text preprocessing** — `transform_text()`: lowercase → tokenize → keep alphanumeric
   tokens → remove English stopwords → Porter stemming.
4. **Feature extraction** — TF-IDF vectorization of the transformed text.
5. **Modelling** — 80/20 train-test split, then Gaussian / Multinomial / Bernoulli Naive Bayes
   and an SVM, evaluated on accuracy and precision.
6. **Inference** — an interactive `classify_sms()` demo that prints spam probability.

## Results

| Model | Features | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Gaussian Naive Bayes | TF-IDF | 0.8762 | 0.523 | 0.819 |
| Multinomial Naive Bayes | TF-IDF | 0.9594 | 1.000 | 0.696 |
| Bernoulli Naive Bayes | TF-IDF | 0.9700 | 0.973 | 0.797 |
| SVC (sigmoid kernel) | TF-IDF | 0.9729 | 0.974 | — |

Recall is computed from the saved confusion matrices (138 spam messages in the test split).

## How to run

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud xgboost joblib
```

Open `sms_spam_pipeline.ipynb` in Jupyter or upload it to
[Google Colab](https://colab.research.google.com/) and run the cells top to bottom.
The notebook downloads the required NLTK corpora (`punkt`, `punkt_tab`, `stopwords`) itself.

## Notes

- Cells that mount Google Drive are Colab-specific; on a local Jupyter install they can be skipped.
- The final cell exports the fitted vectorizer + model as a single `joblib` pickle.
