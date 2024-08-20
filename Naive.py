import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load the data
df = pd.read_excel("/Users/sakshi_admin/Feedback Analysis/Feedback Analysis.xlsx")

# Drop rows with null values in 'QuestionCategory' and handle missing 'RequestText'
df.dropna(subset=['QuestionCategory'], inplace=True)
df['RequestText'].fillna('', inplace=True)  # Fill missing values with empty strings

# Convert 'QuestionCategory' and 'RequestText' to strings
df['QuestionCategory'] = df['QuestionCategory'].astype(str)
df['RequestText'] = df['RequestText'].astype(str)

# Basic text cleaning
df['RequestText'] = df['RequestText'].str.lower()
df['RequestText'] = df['RequestText'].str.replace('[^\w\s]', '', regex=True)

# Feature extraction with n-grams
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=2000)
X = vectorizer.fit_transform(df['RequestText'])
y = df['QuestionCategory']

# Encode the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check class distribution
class_distribution = Counter(y_train)
print("Class distribution:", class_distribution)

# Applying SMOTE
min_samples = min(class_distribution.values())
if min_samples > 1:
    smote = SMOTE(random_state=42, k_neighbors=min(min_samples - 1, 5))
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    y_train_used = y_train_smote
    X_train_used = X_train_smote
else:
    print("Not enough samples in each class for SMOTE, proceeding without SMOTE")
    y_train_used = y_train
    X_train_used = X_train

# Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_used, y_train_used)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Print statistics and evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Classification report
target_names = label_encoder.inverse_transform(np.unique(y_test))
print(classification_report(y_test, y_pred, target_names=target_names))


# Accuracy: 0.6387995712754555
# Precision: 0.027897637059867374
# Recall: 0.016454751966325405
# F1 Score: 0.01662645847731526