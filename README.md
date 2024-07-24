# Automatic Ticket Classification

## Overview
This notebook is designed to perform automatic ticket classification using Natural Language Processing (NLP) techniques. It processes textual data from customer support tickets, applies NLP methods for feature extraction, and builds a machine learning model to categorize tickets into predefined classes.

## Libraries Used

- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical computations.
- **re, string**: Text preprocessing and manipulation.
- **nltk**: Text processing (tokenization, stemming, stopword removal).
- **spaCy**: Advanced text processing tasks.
- **seaborn & matplotlib**: Data visualization.
- **plotly**: Interactive visualizations.
- **scikit-learn**: Machine learning tools (model selection, training, evaluation).
- **TextBlob**: Simplifies text processing tasks (sentiment analysis, noun phrase extraction).

## Workflow

1. **Data Loading and Exploration**
   - Load and explore dataset characteristics.
   
2. **Text Preprocessing** 
   - Tokenization: Splitting text into individual words.
   - Stopword Removal: Eliminating common, non-informative words.
   - Stemming/Lemmatization: Reducing words to their base form.
   
3. **Feature Extraction**
   - Use `CountVectorizer` and `TfidfVectorizer` for numerical feature transformation.
   
4. **Model Building**
   - Split dataset into training and testing sets.
   - Train a classification model (e.g., Random Forest, SVM) with cross-validation.
   - Fine-tune hyperparameters using `RandomizedSearchCV` and `GridSearchCV`.
   
5. **Model Evaluation**
   - Generate confusion matrix and classification report.
   - Calculate precision, recall, and F1 scores.
   - Plot ROC curves to assess performance.

6. **Visualization**
   - Use Seaborn and Plotly to plot data distributions and feature importance.
