# Sentiment Analysis Using Twitter Dataset

## Project Overview

This project focuses on sentiment analysis to classify tweets into positive or negative categories. The significance of this analysis is substantial due to the vast amount of real-time data generated on platforms like Twitter, providing invaluable insights for businesses, policymakers, researchers, and marketers.

## Data Preprocessing

We used the Sentiment140 dataset with approximately 1.6 million tweets. A subset of 100,000 tweets was selected to ensure data variety and volume for effective training. The preprocessing steps included:

- **Stemming and Lemmatization:** Reducing words to their base or dictionary forms to standardize and simplify the dataset.
- **Vectorization:** Converting text into numerical formats using TF-IDF and Count Vectorization for model input.

## Models Overview

Three models were evaluated:

1. **Logistic Regression:** A simple and interpretable model, ideal for baseline comparisons.
2. **Neural Network (MLP):** Flexible and capable of capturing complex patterns, using layers and dropout techniques to prevent overfitting.
3. **DistilBERT:** A streamlined version of BERT, efficient and context-aware, outperforming other models with an accuracy of 83%.

## Results and Model Comparison

- Logistic Regression achieved 74.48% accuracy.
- Neural Network had a lower accuracy of 71.17%.
- DistilBERT significantly outperformed others with an accuracy of 83%.

## User Interface

The user interface was developed using Streamlit, allowing real-time sentiment analysis with a user-friendly experience. This UI enables users to input text and receive sentiment classifications instantly.

## Future Work

Potential enhancements include exploring more complex models like RoBERTa or GPT-3, expanding the dataset, real-time data processing, and integration with other business systems for automated sentiment categorization.

## References

1. Sentiment140 dataset from Kaggle.
2. Various papers and articles on logistic regression, neural networks, and BERT models.
