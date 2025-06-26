# 📧 Spam Detection using Naive Bayes

This is a machine learning project that demonstrates how to build a simple yet powerful **Spam Detection** system using **Multinomial Naive Bayes**, one of the most effective algorithms for text classification problems. This classifier can detect whether an SMS message is spam or not based on its content.

---

## 🧠 Problem Statement

Given a dataset of SMS messages labeled as either `ham` (not spam) or `spam`, the goal is to build a model that can accurately classify new, unseen messages into these two categories.

---

## 📂 Dataset Description

- **Source**: [SMS Spam Collection Dataset](https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv)
- **Format**: Tab-separated file (TSV)
- **Columns**:
  - `label`: Contains either `ham` (not spam) or `spam`
  - `message`: The actual SMS text content

---

## 🛠️ Tools & Libraries Used

- `pandas` – Data manipulation
- `scikit-learn` – For model building and evaluation
- `CountVectorizer` – To convert text data into numerical features (Bag of Words)
- `MultinomialNB` – Naive Bayes classifier suitable for text data
- `train_test_split` – To split the dataset
- `accuracy_score` – To evaluate model performance

---

## 🚀 Step-by-Step Workflow

### 📥 1. Import Libraries

```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
