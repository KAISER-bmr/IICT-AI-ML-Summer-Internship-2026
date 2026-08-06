# =============================================================================
# PROJECT 1: AI-Powered Fake News Detection Using Text Classification
# =============================================================================
# Name   : Prathamesh Pranay Chaumwal
# Roll No: 113460 
# Programme: Summer Internship in AI & ML 2026
# Institute: Indian Institute of Computing and Technology (IICT)
# Instructor: Dr. Ashok Gopalakrishnan
#
# DATASET:  WELFake Dataset
# SOURCE:   https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
# FILE:     WELFake_Dataset.csv
#
# HOW TO RUN:
#   1. Download WELFake_Dataset.csv from Kaggle (link above).
#   2. Place it in the SAME folder as this script / notebook.
#   3. Run all cells top to bottom.
#
# COLUMNS IN DATASET:
#   - serial_num : row index 
#   - title      : article headline
#   - text       : article body (main feature)
#   - label      : 0 = Fake, 1 = Real
# =============================================================================

# =============================================================================
# IMPORTS
# =============================================================================

# Pandas - Used to load and work with datasets (tables like Excel sheets)
import pandas as pd
# NumPy - Helps perform fast mathematical and numerical operations
import numpy as np
# re (Regular Expressions) - Used to clean text (remove symbols, numbers, etc.)
import re
# os - Helps access files and folders on the computer
import os
# warnings - Used to hide unnecessary warning messages
import warnings
warnings.filterwarnings('ignore') # Ignore all warning messages

# =============================================================================
# NLTK (Natural Language Toolkit)
# Used for text preprocessing in Machine Learning
# =============================================================================

import nltk

# These will download the required language resources (runs only once if already downloaded)
nltk.download('stopwords', quiet = True) # Common words like "the", "is", "and"
nltk.download('wordnet', quiet = True)  # Dictionary used for lemmatization
nltk.download('punkt', quiet = True) # Tokenizer used to split text into words

# Stopwords - Removes common words that usually don't add much meaning
from nltk.corpus import stopwords 

# Lemmatizer - Converts words to their base form (e.g., "running" → "run")
from nltk.stem import WordNetLemmatizer

# =============================================================================
# Scikit-Learn (Machine Learning Library)
# =============================================================================

# Converts text into numerical features that ML models can understand
from sklearn.feature_extraction.text import TfidfVectorizer
# Splits data into training/testing sets and performs cross-validation
from sklearn.model_selection import train_test_split, cross_val_score
# K-Nearest Neighbors (KNN) classification model
from sklearn.neighbors import KNeighborsClassifier
# Logistic Regression classification model
from sklearn.linear_model import LogisticRegression
# Random Forest classification model
from sklearn.ensemble import RandomForestClassifier
# Multi-Layer Perceptron (Basic Neural Network) classification model
from sklearn.neural_network import MLPClassifier

# Tools used to evaluate how well the ML model performs
from sklearn.metrics import (
    accuracy_score,             # Calculates prediction accuracy
    classification_report,      # Shows Precision, Recall & F1-Score
    confusion_matrix,           # Creates confusion matrix values
    ConfusionMatrixDisplay,     # Displays confusion matrix as a graph
    roc_auc_score,              # Calculates AUC score
    roc_curve                   # Generates ROC curve data
)

# =============================================================================
# Data Visualization
# =============================================================================

# Main plotting library
import matplotlib
# Makes graphs clearer by increasing image resolution
matplotlib.rcParams['figure.dpi'] = 120
# Used to create graphs and plots
import matplotlib.pyplot as plt
# Creates more attractive statistical graphs
import seaborn as sns

# =============================================================================
# Other Utility
# =============================================================================

# Counts how many times each item appears (useful for word frequencies)
from collections import Counter


# Confirmation message
print("All imports successful.")