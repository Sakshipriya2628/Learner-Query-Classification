# Key Changes & Rationale:
#This approach will involve revisiting the data preprocessing to ensure it's optimal and experimenting with a different model, 
# such as a Support Vector Machine (SVM)
#, which is often very effective for text classification tasks due to its capability to handle high-dimensional spaces 
# and identify complex patterns.
# Model Choice: Switched to a linear Support Vector Machine (SVM). SVMs are particularly good for text classification due to their 
#effectiveness in high-dimensional spaces, which is typical for text data.
# Preprocessing: Continued focus on cleaning and normalizing text data to prepare it effectively for modeling.
# Vectorization: Kept to 500 features in TF-IDF to manage complexity and focus on the most informative features.
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

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
df = pd.read_excel('/Users/sakshi_admin/Documents/Projects Backup/Feedback Analysis/Feedback Analysis.xlsx')
df['RequestText'] = df['RequestText'].fillna('missing')  # Handle missing values
df['QuestionCategory'] = df['QuestionCategory'].astype(str)  # Ensure correct data type

# Apply preprocessing
df['processed_text'] = df['RequestText'].apply(preprocess_text)

# Vectorization with TF-IDF
vectorizer = TfidfVectorizer(max_features=500)  # Limiting to 500 features for simplicity
X = vectorizer.fit_transform(df['processed_text'])
y = df['QuestionCategory']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using SVM for classification
svc_model = SVC(kernel='linear')  # Using linear kernel
svc_model.fit(X_train, y_train)

# Predict on the test data
predictions = svc_model.predict(X_test)

# Evaluate the model using classification report and accuracy score
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))
