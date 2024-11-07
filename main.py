from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Load your dataset
with open('classification.json', 'r') as f:
    data = json.load(f)

# Prepare the dataset: Extract 'text' and 'label' fields for training
def prepare_data(data):
    texts = []
    labels = []

    for entry in data:
        texts.append(entry['text'])
        labels.append(entry['label'])  # Labels are already in 0, 1, or 2 format

    return texts, labels

texts, labels = prepare_data(data)

# Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)

# Tokenize the dataset
def tokenize_function(texts):
    return tokenizer(texts, padding="max_length", truncation=True, return_tensors='pt')

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Convert encodings to Dataset format
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'], 
    'attention_mask': train_encodings['attention_mask'], 
    'labels': train_labels
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'], 
    'attention_mask': val_encodings['attention_mask'], 
    'labels': val_labels
})

# Initialize the model and move it to the appropriate device
model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels=3).to(device)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Adjust if needed
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=1,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
    fp16=True  # Enable mixed precision if GPU is available
)

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')

# Example function to load and predict using the trained model
def predict(input_text):
    model = BertForSequenceClassification.from_pretrained('./saved_model').to(device)
    tokenizer = BertTokenizer.from_pretrained('./saved_model')

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True).to(device)
    outputs = model(**inputs)

    # Get the predicted class (Buy, Sell, Hold)
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    
    # Convert class label back to a string
    label_map = {0: "Buy", 1: "Sell", 2: "Hold"}
    return label_map[predicted_class]

# Example usage of the trained model
test_text = "Price: 172.63, MAS: 173.42, MAL: 178.48"
predicted_response = predict(test_text)

print(predicted_response)  # Should print "Buy", "Sell", or "Hold"
