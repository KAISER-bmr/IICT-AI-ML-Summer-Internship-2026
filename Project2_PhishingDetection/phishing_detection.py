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
df = pd.read_csv("Phishing_Email.csv")

# Removes any unnecessary unnamed index column if it exists
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)], errors='ignore')

# Displays the dataset size, column names and first few rows
print(f"\nDataset loaded: {df.shape[0]:,} emails, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 3 rows:\n{df.head(3)}")

# ── 1.2 Rename columns for consistency ───────────────────────────────────────

# Renames the original dataset columns to simpler names
# The dataset uses 'Email Text' and 'Email Type'
df = df.rename(columns={
    'Email Text': 'text',
    'Email Type': 'label_raw'
})

# ── 1.3 Check missing values ─────────────────────────────────────────────────

# Checks how many missing values are present in each column
print(f"\nMissing values:\n{df.isnull().sum()}")

# Replaces missing email text with an empty string
df['text'] = df['text'].fillna('')

# ── 1.4 Encode labels ────────────────────────────────────────────────────────

# Converts text labels into numerical labels for Machine Learning
# "Phishing Email" -> 1, "Safe Email" -> 0
df['label'] = (df['label_raw'] == 'Phishing Email').astype(int)

# Displays the number of phishing and safe emails
print(f"\nLabel distribution:")
print(df['label_raw'].value_counts())

# Shows the encoded label counts
print(f"\nEncoded: Phishing=1 ({df['label'].sum():,}) | "
      f"Safe=0 ({(df['label']==0).sum():,})")

# Plot label distribution

# Creates a bar graph showing phishing and safe email counts
plt.figure(figsize=(5, 4))
df['label_raw'].value_counts().plot(
    kind='bar', color=['#e74c3c', '#2ecc71'], edgecolor='black'
)

# Adds title and axis labels to the graph
plt.title('Label Distribution: Phishing vs Safe Emails')
plt.xlabel('Email Type')
plt.ylabel('Count')

# Adjusts the label angle for better readability
plt.xticks(rotation=15, ha='right')
plt.tight_layout()

# Saves the graph as an image file
plt.savefig('p2_plot_01_label_distribution.png')
plt.show()
print("Saved: p2_plot_01_label_distribution.png")

# ── 1.5 Metadata feature extraction ──────────────────────────────────────────

# Extracts structural features before cleaning the text
# This is the key addition over Project 1: structural features extracted
# BEFORE cleaning so URLs and punctuation are still present.

def extract_metadata(text):
    """
    Extract structural metadata from raw email text.
    These features capture suspicious STRUCTURE (not just language).
    """

    # Converts the input into a string to avoid datatype errors
    text = str(text)

    # Counts the number of URLs present in the email
    url_count = len(re.findall(r'https?://\S+|www\.\S+', text))

    # Counts words commonly associated with urgency or phishing attempts
    urgent_words = len(re.findall(
        r'\b(urgent|immediately|now|click|verify|suspend|expire|warning|'
        r'alert|limited|confirm|account|password|bank|prize|winner|free|'
        r'offer|act|login|update|security|billing|payment)\b',
        text.lower()
    ))

    # Counts exclamation marks in the email
    exclaim_count = text.count('!')

    # Calculates the proportion of uppercase characters
    caps_ratio = (sum(1 for c in text if c.isupper())
                  / max(len(text), 1))

    # Stores whether the email contains at least one URL
    has_url = int(url_count > 0)

    # Stores whether the email contains a dollar sign
    has_dollar = int('$' in text)

    # Returns all extracted metadata features
    return [url_count, urgent_words, exclaim_count,
            caps_ratio, has_url, has_dollar]


# Extracts metadata features from every email
print("\nExtracting metadata features...")
meta_raw = np.array([extract_metadata(t) for t in df['text']])

# Names given to each extracted metadata feature
meta_labels = ['URL Count', 'Urgent Words', 'Exclamations',
               'Caps Ratio', 'Has URL', 'Has Dollar Sign']

# Displays the size of the metadata feature matrix
print(f"Metadata matrix shape: {meta_raw.shape}")

# Converts metadata into a DataFrame for easier analysis
print(f"\nMean metadata values by class:")
meta_df = pd.DataFrame(meta_raw, columns=meta_labels)

# Adds the phishing/safe label to the metadata DataFrame
meta_df['label'] = df['label'].values

# Calculates the average metadata values for each class
print(meta_df.groupby('label').mean().rename(index={0:'Safe',1:'Phishing'}))

# Plot metadata comparison

# Calculates average metadata values for phishing emails
phishing_meta = meta_raw[df['label']==1].mean(axis=0)

# Calculates average metadata values for safe emails
safe_meta = meta_raw[df['label']==0].mean(axis=0)

# Creates positions for each metadata feature on the graph
x = np.arange(len(meta_labels))

# Creates a grouped bar chart for comparison
fig, ax = plt.subplots(figsize=(10, 5))

# Plots average metadata values for phishing emails
b1 = ax.bar(x - 0.2, phishing_meta, 0.4, label='Phishing',
            color='#e74c3c', edgecolor='black')

# Plots average metadata values for safe emails
b2 = ax.bar(x + 0.2, safe_meta, 0.4, label='Safe',
            color='#2ecc71', edgecolor='black')

# Adds feature names and graph labels
ax.set_xticks(x)
ax.set_xticklabels(meta_labels, rotation=15, ha='right')
ax.set_title('Metadata Feature Comparison: Phishing vs Safe Emails')
ax.set_ylabel('Mean Value')
ax.legend()

# Displays the numerical value above each bar
ax.bar_label(b1, fmt='%.2f', padding=2, fontsize=8)
ax.bar_label(b2, fmt='%.2f', padding=2, fontsize=8)

# Adjusts spacing and saves the graph
plt.tight_layout()
plt.savefig('p2_plot_02_metadata_comparison.png')
plt.show()
print("Saved: p2_plot_02_metadata_comparison.png")