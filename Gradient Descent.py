import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    text = str(text)  # Ensure text is treated as a string
    text = text.lower()  # Convert to lower case
    tokens = word_tokenize(text)  # Tokenize
    tokens = [token for token in tokens if token.isalnum()]  # Keep alphanumeric words
    stop_words_set = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words_set]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize
    return ' '.join(tokens)

# Load data
df = pd.read_excel('/Users/sakshi_admin/Feedback Analysis/Feedback Analysis.xlsx')

# Drop rows with null values in both 'RequestText' and 'QuestionCategory' columns
df = df.dropna(subset=['RequestText', 'QuestionCategory'])

df['QuestionCategory'] = df['QuestionCategory'].astype(str)  # Ensure correct data type

# Apply preprocessing
df['processed_text'] = df['RequestText'].apply(preprocess_text)

# Vectorization with TF-IDF
vectorizer = TfidfVectorizer(max_features=500)  # Limiting to 500 features for simplicity
X = vectorizer.fit_transform(df['processed_text'])
y = df['QuestionCategory']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using SGDClassifier for SVM with hinge loss and parameter tuning
svm_model = make_pipeline(StandardScaler(with_mean=False), SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, alpha=0.0001, random_state=42, verbose=1))
svm_model.fit(X_train, y_train)

# Predict on the test data
predictions = svm_model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))

# There is no built-in loss curve for SGDClassifier to plot directly.
# If you need a custom plot, you'd need to implement a callback or similar mechanism to track loss per iteration.
