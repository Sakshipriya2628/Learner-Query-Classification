import pandas as pd
import spacy
from spacy.training import Example
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the dataset
data = pd.read_excel("/Users/sakshi_admin/Feedback Analysis/Feedback Analysis.xlsx")

# Drop rows with null values in 'RequestText' or 'QuestionCategory'
data.dropna(subset=['RequestText', 'QuestionCategory'], inplace=True)

# Ensure all data in 'RequestText' is treated as string
data['RequestText'] = data['RequestText'].astype(str).str.lower().str.strip()

# Convert categories to string to ensure compatibility with SpaCy
data['QuestionCategory'] = data['QuestionCategory'].astype(str)

# Splitting the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Load SpaCy model
nlp = spacy.blank("en")

# Add text classifier to the pipeline with appropriate configuration
config = {
    "threshold": 0.5,
    "model": {
        "@architectures": "spacy.TextCatBOW.v2",
        "exclusive_classes": True,
        "ngram_size": 1,
        "no_output_layer": False
    }
}
textcat = nlp.add_pipe("textcat", config=config)

# Add labels to text classifier
for category in train_data['QuestionCategory'].unique():
    textcat.add_label(str(category))  # Ensure the label is a string

# Training the model
train_texts = train_data['RequestText'].values

# Correctly creating a 'cats' dictionary for each text example
train_labels = []
for cat in train_data['QuestionCategory']:
    cat_dict = {label: label == cat for label in textcat.labels}
    train_labels.append({'cats': cat_dict})

train_examples = [Example.from_dict(nlp.make_doc(text), cats) for text, cats in zip(train_texts, train_labels)]

optimizer = nlp.initialize()

for i in range(10):
    losses = {}
    random.shuffle(train_examples)
    for batch in spacy.util.minibatch(train_examples, size=8):
        nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i}: {losses}")

# Predicting on the test set
test_texts = test_data['RequestText'].values
test_cats = test_data['QuestionCategory'].values

predictions = []
for text in test_texts:
    doc = nlp(text)
    predictions.append(max(doc.cats, key=doc.cats.get))

# Evaluation metrics
accuracy = accuracy_score(test_cats, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(test_cats, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Accuracy: 0.6784565916398714
# Precision: 0.6134506996123148
# Recall: 0.6784565916398714
# F1 Score: 0.6320719027871641