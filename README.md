# FakeNewDetection

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Manual Testing](#manual-testing)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Project Structure](#project-structure)

## Introduction

In the era of information overload, distinguishing between fake and real news has become increasingly challenging yet crucial. This project leverages machine learning techniques to build a Fake News Detection System. By utilizing TF-IDF vectorization and various classification algorithms, the system effectively classifies news articles as fake or genuine.

## Features

- **Data Preprocessing:** Cleans and normalizes text data by removing URLs, special characters, stopwords, and applying lemmatization.
- **Visualization:** Generates word clouds to visualize the most frequent words in fake and real news.
- **Multiple Classifiers:** Implements and evaluates multiple machine learning models, including Multinomial Naive Bayes, Random Forest, Logistic Regression, and Linear SVM.
- **Manual Testing:** Allows users to input custom news articles and see predictions from all models.
- **Performance Evaluation:** Provides accuracy, precision, recall, F1 score, and confusion matrices for each model.

## Dataset

The dataset comprises 72,134 articles focusing on current events in the United States. These articles are categorized into two main groups: 37,106 real news articles and 35,028 fake news articles. The data has been aggregated from reputable sources such as Kaggle, McIntire, Reuters, and BuzzFeed Political. These sources provide a substantial volume of text, which not only helps mitigate overfitting but also enhances the classification performance of machine learning algorithms. This dataset is a valuable resource for training models to effectively detect fake news, thereby reducing the impact of misinformation.

The dataset is structured into four main features:

- **Serial number:** The index of each article, starting from 0
- **Title:** The headline of the article
- **Text:** The detailed content of the article
- **Label:** Classification label, indicating whether the article is real or fake [0 for real news and 1 for fake news]

## Manual Testing

After running the script, you will be prompted to enter a news article for classification. The system will display predictions from all implemented models.

**Example:**

```diff
===== Fake News Detection System =====

Text: Your news article here...
Output:
```

```sql
Predictions from models:
    Model                Result
0   Naive Bayes          Fake New
1   Random Forest        Fake New
2   Logistic Regression  Fake New
3   Linear SVM           Fake New
```

## Models Used

1. Multinomial Naive Bayes
2. Random Forest Classifier
3. Logistic Regression
4. Linear SVM

Each model is trained on the TF-IDF vectorized data and evaluated using standard classification metrics.

## Evaluation Metrics

The following metrics are used to evaluate the performance of each model:

- Accuracy: The proportion of correctly classified instances.
- Precision: The proportion of positive identifications that were actually correct.
- Recall: The proportion of actual positives that were identified correctly.
- F1 Score: The harmonic mean of precision and recall.
- Confusion Matrix: A table used to describe the performance of the classification model.

## Results

**Multinomial Naive Bayes**
```yaml
Evaluted the model Multinomial Naive Bayes
Accuracy: 0.851010
Precision: 0.820926
Recall: 0.854395
F1 Score: 0.837326
Confusion Matrix:
 [[5903  1056]
 [ 825 4841]]
 ```

**Random Forest**
```yaml
Evaluted the model Random Forest
Accuracy: 0.920871
Precision: 0.907883
Recall: 0.916696
F1 Score: 0.912268
Confusion Matrix:
 [[6432  527]
 [ 472  5194]]
 ```

**Logistic Regression**
```yaml
Evaluted the model Logistic Regression
Accuracy: 0.921030
Precision: 0.897226
Recall: 0.930639
F1 Score: 0.913627
Confusion Matrix:
 [[6355   604]
 [ 393  5273]]
 ```

**Linear SVM**
```yaml
Evaluted the model SGD Classifier
Accuracy: 0.946931
Precision: 0.933831
Recall: 0.948994
F1 Score: 0.941352
Confusion Matrix:
 [[6578   381]
 [ 289 5377]]
```

**Observation:**

- Random Forest and Linear SVM exhibit the highest accuracy, precision, recall, and F1 scores, indicating superior performance in classifying fake and real news.
- Multinomial Naive Bayes also performs well but still lower than other models.
- Logistic Regression demonstrates robust performance with high recall and F1 scores.

## Project Structure

```python
fake-news-detection/
├── Data/
│   └── WELFake_Dataset.zip
├── FakeNewsDetection.ipynb
├── README.md
```

- Data/: Contains the zipped dataset file.
- FakeNewsDetection.ipynb: The Jupyter Notebook for data preprocessing, model training, evaluation, and manual testing.
- README.md: This documentation file.