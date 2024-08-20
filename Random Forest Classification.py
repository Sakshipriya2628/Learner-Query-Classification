import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    # Convert to string in case of numeric values
    text = str(text)
    text = text.lower()  # Lowercase text
    words = word_tokenize(text)  # Tokenize into words
    words = [word for word in words if word.isalnum()]  # Remove non-alphanumeric characters
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
    return ' '.join(words)

# Load data from an Excel file
df = pd.read_excel('/Users/sakshi_admin/Documents/Projects Backup/Feedback Analysis/Feedback Analysis.xlsx')

# Ensure all data in the target column is of type string
df['QuestionCategory'] = df['QuestionCategory'].astype(str)

# Fill missing values in the text column
df['RequestText'] = df['RequestText'].fillna('missing')

# Preprocess the text data
df['processed_text'] = df['RequestText'].apply(preprocess_text)

# Vectorize the processed text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['QuestionCategory']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForest classifier
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Predict on the test data
predictions = classifier.predict(X_test)

# Evaluate the model using classification report and accuracy score
print("Classification Report:")
print(classification_report(y_test, predictions))
print("Accuracy Score:")
print(accuracy_score(y_test, predictions))


# def categorize_new_question(new_question):
#     processed = preprocess_text(new_question)
#     vectorized = vectorizer.transform([processed])
#     category_prediction = classifier.predict(vectorized)
#     return category_prediction[0]

# # Example usage
# new_question = "How do I calculate the area of a circle?"
# predicted_category = categorize_new_question(new_question)
# print(f"The question '{new_question}' belongs to the category: {predicted_category}")

