# =============================================================================
# PROJECT 2: AI-Driven Phishing Email Detection Using NLP
# =============================================================================
# Name   : Prathamesh Pranay Chaumwal
# Roll No: 113460
# Programme: Summer Internship in AI & ML 2026
# Institute: Indian Institute of Computing and Technology (IICT)
# Instructor: Dr. Ashok Gopalakrishnan
#
# DATASET:  Phishing Email Detection Dataset
# SOURCE:   https://www.kaggle.com/datasets/subhajournal/phishingemails
# FILE:     Phishing_Email.csv
#
# HOW TO RUN:
#   1. Download Phishing_Email.csv from Kaggle (link above).
#   2. Place it in the SAME folder as this script / notebook.
#   3. Run all cells top to bottom.
#
# COLUMNS IN DATASET:
#   - Email Text : raw email body (main feature)
#   - Email Type : "Phishing Email" or "Safe Email" (target)
#
# KEY DIFFERENCE FROM PROJECT 1:
#   This pipeline extracts BOTH text features (TF-IDF) AND structural
#   metadata features (URL count, urgency word count, exclamation marks,
#   caps ratio) — then combines them into a single feature matrix.
#   This reflects the real-world reality that phishing emails have both
#   suspicious LANGUAGE and suspicious STRUCTURE.
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

# Pandas - Used to load and work with datasets
import pandas as pd
# NumPy - Used for numerical and mathematical operations
import numpy as np
# re - Used for text cleaning with Regular Expressions
import re
# Warnings - Used to hide unnecessary warning messages
import warnings
warnings.filterwarnings('ignore')  # Ignore warning messages

# =============================================================================
# NLTK (Natural Language Toolkit)
# Used for text preprocessing
# =============================================================================

import nltk

# Downloads the required NLTK resources for text preprocessing
nltk.download('stopwords', quiet=True)  # Common words like "the", "is", "and"
nltk.download('wordnet',   quiet=True)  # Dictionary used for lemmatization
nltk.download('punkt',     quiet=True)  # Used to split text into words

# Stopwords - Removes common words that usually add little meaning
from nltk.corpus import stopwords
# Lemmatizer - Converts words into their base form
from nltk.stem  import WordNetLemmatizer

# =============================================================================
# Scikit-Learn (Machine Learning Library)
# =============================================================================

# Converts text into numerical TF-IDF features for ML models
from sklearn.feature_extraction.text import TfidfVectorizer
# Splits data and performs cross-validation
from sklearn.model_selection import train_test_split, cross_val_score
# Logistic Regression classification model
from sklearn.linear_model  import LogisticRegression
# Random Forest classification model
from sklearn.ensemble import RandomForestClassifier
# Multinomial Naive Bayes classification model
from sklearn.naive_bayes import MultinomialNB
# Multi-Layer Perceptron neural network classification model
from sklearn.neural_network import MLPClassifier
# Tools used for encoding labels and scaling numerical features
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MaxAbsScaler
# Tools used to evaluate model performance
from sklearn.metrics                 (
    accuracy_score,                  # Calculates prediction accuracy
    classification_report,           # Shows Precision, Recall and F1-Score
    confusion_matrix,                # Creates confusion matrix values
    ConfusionMatrixDisplay,          # Displays confusion matrix as a graph
    roc_auc_score,                   # Calculates AUC score
    roc_curve                         # Generates ROC curve data
)
# Combines text features with additional metadata features
from scipy.sparse import hstack, csr_matrix

# =============================================================================
# DATA VISUALIZATION
# =============================================================================

# Main plotting library
import matplotlib
# Increases the resolution of generated graphs
matplotlib.rcParams['figure.dpi'] = 120
# Used to create graphs and plots
import matplotlib.pyplot as plt
# Used to create statistical and attractive visualizations
import seaborn as sns
# Counts how frequently items such as words appear
from collections import Counter


# Confirmation messages
print("All imports successful.")
print("Note: This pipeline uses TF-IDF + metadata features (combined matrix).")

# =============================================================================
# WEEK 1: DATA COLLECTION & CLEANING
# =============================================================================

# Displays the Week 1 heading in the output
print("\n" + "="*60)
print("WEEK 1: DATA LOADING & TEXT CLEANING")
print("="*60)

# ── 1.1 Load Dataset ─────────────────────────────────────────────────────────

# Loads the phishing email dataset into a Pandas DataFrame