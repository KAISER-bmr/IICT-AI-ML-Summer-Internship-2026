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