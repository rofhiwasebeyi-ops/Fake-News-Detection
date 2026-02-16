import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve)
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load the dataset
df = pd.read_csv('data/preprocessed/preprocessed_dataset.csv')

# Filter out any rows with missing or empty 'clean_text'
df = df[df['clean_content'].notna() & (df['clean_content'].str.strip() != '')]

# Validate labels in pandas
if not df['label'].isin([0, 1]).all():
    raise ValueError("Labels must be only 0 or 1")

# Split the dataframe into train and test
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['label'])

# Calculate class weights from train labels
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=train_df['label'].values)
weights = torch.tensor(class_weights_array, dtype=torch.float).to(device)
print(f"Class weights: {class_weights_array}")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')


# Tokenize function that works with pandas Series
def tokenize_texts(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')


# Tokenize train and test datasets
train_encodings = tokenize_texts(train_df['clean_content'])
test_encodings = tokenize_texts(test_df['clean_content'])

# Convert labels to tensors
train_labels = torch.tensor(train_df['label'].values).to(device)
test_labels = torch.tensor(test_df['label'].values).to(device)


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encodings, train_labels)
test_dataset = NewsDataset(test_encodings, test_labels)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2).to(device)


# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}


# Custom Trainer subclass to use class weights in loss
class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        labels = labels.to(model.device)  # ensure labels are on the same device
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.tensor([1.0, 2.0]).to(model.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)


# Trainer
trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train(resume_from_checkpoint=True)

# Evaluate on test set
model.eval()
all_logits = []
batch_size = 8
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i+batch_size]
    inputs = {k: v for k, v in batch[0].items() if k != 'labels'}
    # Unsqueeze dims if necessary and move tensors to model device
    inputs = {k: v.unsqueeze(0) if v.ndim ==
              1 else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu())

logits = torch.cat(all_logits, dim=0).numpy()
preds = logits.argmax(-1)
probs = torch.softmax(torch.tensor(logits), dim=1)[
    :, 1].numpy()  # probability of class 1
y_true = test_labels.cpu().numpy()

# Compute metrics
accuracy = accuracy_score(y_true, preds)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, preds, average='binary')
roc_auc = roc_auc_score(y_true, probs)

print("\nDistilBERT Evaluation Metrics")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

# Confusion matrix
cm = confusion_matrix(y_true, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues")
plt.title("DistilBERT Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(y_true, preds))

# ROC curve
fpr, tpr, _ = roc_curve(y_true, probs)
plt.plot(fpr, tpr, label=f"DistilBERT (AUC={roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - DistilBERT")
plt.legend()
plt.show()

# Save model and tokenizer
model.save_pretrained('models/fine_tuned_distilbert_fake_news')
tokenizer.save_pretrained('models/fine_tuned_distilbert_fake_news')
