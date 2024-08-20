import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load data
df = pd.read_excel('/Users/sakshi_admin/Feedback Analysis/Feedback Analysis.xlsx')
df['RequestText'] = df['RequestText'].fillna('missing')  # Fill missing values
df['RequestText'] = df['RequestText'].astype(str)  # Ensure all text is in string format

# Encode labels and remove classes with fewer than 2 instances
labels, _ = pd.factorize(df['QuestionCategory'])
df['encoded_labels'] = labels
class_counts = df['encoded_labels'].value_counts()
df = df[df['encoded_labels'].isin(class_counts[class_counts >= 2].index)]

# Re-factorize labels after filtering
df['encoded_labels'], _ = pd.factorize(df['encoded_labels'])

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def encode_data(tokenizer, questions, max_length):
    input_ids, attention_masks = [], []
    for question in questions:
        encoded = tokenizer.encode_plus(
            text=question,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.tensor(input_ids), torch.tensor(attention_masks)

# Encode data
input_ids, attention_masks = encode_data(tokenizer, df['RequestText'], max_length=128)
labels = torch.tensor(df['encoded_labels'].values)

# Train-test split
train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, stratify=labels, random_state=42, test_size=0.1)
train_masks, val_masks, _, _ = train_test_split(attention_masks, labels, stratify=labels, random_state=42, test_size=0.1)

# DataLoaders
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)
validation_data = TensorDataset(val_inputs, val_masks, val_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=32)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(df['encoded_labels'].unique())
)
model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Function to calculate accuracy
def flat_accuracy(preds, labels):
    pred_flat = torch.argmax(preds, dim=1).flatten()
    labels_flat = labels.flatten()
    return torch.sum(pred_flat == labels_flat) / len(labels_flat)

# Training loop
model.train()
for epoch in range(4):
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(model.device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Training Loss: {total_loss / len(train_dataloader)}")

# Validation
model.eval()
eval_accuracy, eval_steps = 0, 0
for batch in validation_dataloader:
    batch = tuple(t.to(model.device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    eval_accuracy += flat_accuracy(outputs[0], b_labels)
    eval_steps += 1
print(f"Validation Accuracy: {eval_accuracy / eval_steps}")

# Epoch 1, Training Loss: 1.5009887120225927
# Epoch 2, Training Loss: 1.040264870439257
# Epoch 3, Training Loss: 0.8439414206441942
# Epoch 4, Training Loss: 0.6772469442624313
# Validation Accuracy: 0.7244318127632141