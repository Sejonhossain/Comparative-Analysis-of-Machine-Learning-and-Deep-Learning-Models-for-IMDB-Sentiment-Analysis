# Comparative Analysis of Machine Learning and Deep Learning Models for IMDB Sentiment Analysis

Comparative benchmarking of classical Machine Learning and Deep Learning models for binary sentiment classification on movie reviews.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Demo / Results](#demo--results)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Models](#models)

## About

This project benchmarks six models — three classical ML algorithms and three recurrent deep learning architectures — for classifying movie reviews as positive or negative. It compares them not just on accuracy, but on training time and efficiency, to explore whether deep learning is actually worth the extra computational cost for this kind of task.

## Features

- Clean, reusable text preprocessing pipeline
- Side-by-side ML and DL model training
- Accuracy, precision, recall, and F1 score comparison
- Confusion matrices and comparison bar charts
- Training time / efficiency benchmarking
- Modular code — easy to add new models

## Demo / Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 89.24% | 88.22% | 90.77% | 89.48% |
| Multinomial NB | 85.56% | 85.73% | 85.59% | 85.66% |
| Bernoulli NB | 85.03% | 86.08% | 83.84% | 84.95% |
| SimpleRNN | 80.11% | 81.47% | 78.35% | 79.88% |
| LSTM | 87.12% | 91.22% | 82.35% | 86.57% |
| GRU | 88.33% | 89.08% | 87.57% | 88.32% |

Logistic Regression achieves the highest accuracy overall while training significantly faster than any of the deep learning models.

## Tech Stack

- Python 3
- scikit-learn
- TensorFlow / Keras
- NLTK
- pandas, NumPy
- matplotlib, seaborn

## Dataset

IMDB Movie Review Dataset — 50,000 reviews, evenly split between positive and negative sentiment. Available on [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Models

**Machine Learning** (TF-IDF features)
- Logistic Regression
- Multinomial Naive Bayes
- Bernoulli Naive Bayes

**Deep Learning** (tokenized, padded sequences)
- SimpleRNN
- LSTM
- GRU

Each DL model follows: `Embedding → Recurrent Layer → Dropout → Dense (sigmoid)`
