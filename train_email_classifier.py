import pandas as pd
import re
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import multiprocessing
import pickle
import json
import os
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import Dataset
import evaluate

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

def clean_text(text):
    """Enhanced text cleaning specifically for email data."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    # Remove email addresses and URLs
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize text with truncation and max length."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute accuracy and F1 variants."""
    accuracy_metric = evaluate.load("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_metric.compute(predictions=predictions, references=labels)['accuracy'],
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0),
        'f1_weighted': f1_score(labels, predictions, average='weighted', zero_division=0),
        'f1_micro': f1_score(labels, predictions, average='micro', zero_division=0)
    }

class WeightedTrainer(Trainer):
    """Trainer with weighted loss for class imbalance."""
    def __init__(self, class_weights=None, processing_class=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.processing_class = processing_class  # Store tokenizer as processing_class

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def analyze_dataset(df):
    """Analyze dataset statistics."""
    print("=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"Total samples: {len(df):,}")
    print(f"Unique labels: {df['label'].nunique()}")
    print("\nLabel Distribution:")
    label_counts = df['label'].value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({(count / len(df) * 100):.1f}%)")
    df['text_length'] = df['text'].str.len()
    print("\nText Length Statistics:")
    print(f"  Mean: {df['text_length'].mean():.0f} chars")
    print(f"  Median: {df['text_length'].median():.0f} chars")
    print(f"  Min: {df['text_length'].min()} chars")
    print(f"  Max: {df['text_length'].max()} chars")
    imbalance_ratio = label_counts.max() / label_counts.min()
    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 3:
        print("  ⚠️ High imbalance - using weighted training")
    return label_counts

def create_visualizations(df, label_encoder, y_true, y_pred, save_dir):
    """Generate dataset and confusion matrix visualizations."""
    os.makedirs(save_dir, exist_ok=True)
    # Label distribution
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    df['label'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Label Distribution')
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    # Text length distribution
    plt.subplot(1, 2, 2)
    df['text_length'] = df['text'].str.len()
    plt.hist(df['text_length'], bins=50, color='lightgreen', alpha=0.7)
    plt.title('Text Length Distribution')
    plt.xlabel('Length (chars)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/dataset_analysis.png', dpi=300, bbox_inches='tight')
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_percent, annot=True, fmt='.2%', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, cmap="Blues", cbar_kws={'label': 'Percentage'})
    plt.title(f"Confusion Matrix ({len(label_encoder.classes_)}-Class, Normalized)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')

def prepare_data():
    """Load, clean, and split data."""
    try:
        df = pd.read_csv("email_training_data_final.csv")
    except FileNotFoundError:
        raise FileNotFoundError("email_training_data.csv not found! Run create_csv.py first.")
    df.drop_duplicates(subset="text", inplace=True)
    df.dropna(subset=["text", "label"], inplace=True)
    df = df[df['text'].str.len() >= 20]
    analyze_dataset(df)
    df["clean_text"] = df["text"].apply(clean_text)
    MIN_LENGTH, MAX_LENGTH = 15, 2000
    df["text_length"] = df["clean_text"].str.len()
    df = df[(df["text_length"] >= MIN_LENGTH) & (df["text_length"] <= MAX_LENGTH)]
    label_encoder = LabelEncoder()
    df["label_id"] = label_encoder.fit_transform(df["label"])
    num_labels = len(label_encoder.classes_)
    print(f"\nLabel Mapping ({num_labels} classes):")
    for idx, label in enumerate(label_encoder.classes_):
        count = len(df[df['label'] == label])
        print(f"  {idx}: {label} ({count:,} samples)")
    # Check low-sample classes
    for label in df['label'].unique():
        if len(df[df['label'] == label]) < 50:
            print(f"⚠️ {label} has few samples")
    # Stratified split
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.125, stratify=train_df["label_id"], random_state=42)
    except ValueError:
        print("Falling back to random split...")
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=42)
    print(f"Splits: Train {len(train_df):,}, Val {len(val_df):,}, Test {len(test_df):,}")
    return train_df, val_df, test_df, label_encoder, num_labels

def setup_datasets_and_model(train_df, val_df, test_df, num_labels):
    """Convert to datasets, tokenize, and load DistilBERT model with optimized max_length."""
    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label_id"]].reset_index(drop=True)).rename_column("clean_text", "text").rename_column("label_id", "labels")
    val_dataset = Dataset.from_pandas(val_df[["clean_text", "label_id"]].reset_index(drop=True)).rename_column("clean_text", "text").rename_column("label_id", "labels")
    test_dataset = Dataset.from_pandas(test_df[["clean_text", "label_id"]].reset_index(drop=True)).rename_column("clean_text", "text").rename_column("label_id", "labels")
    MODEL_NAME = "distilbert-base-uncased"
    MAX_LENGTH = 400  # Increased to capture more context
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
    # Batched tokenization
    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer, MAX_LENGTH), batched=True)
    for ds in [train_dataset, val_dataset, test_dataset]:
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_dataset, val_dataset, test_dataset, tokenizer, model

def setup_training(train_dataset, val_dataset, train_df, model, tokenizer, num_labels):
    """Compute weights, set args, and initialize trainer with optimized parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label_id']), y=train_df['label_id'])
    # Adjusted weights to reduce confusion
    class_weights_adjusted = torch.tensor(class_weights, dtype=torch.float).clone()
    class_weights_adjusted[1] *= 1.7  # Personal, retained
    class_weights_adjusted[3] *= 1.65  # Spam, slightly reduced
    class_weights_adjusted[5] *= 1.75  # Updates, increased to boost recall
    class_weights = class_weights_adjusted.to(device)
    print(f"Adjusted class weights: {[f'{w:.3f}' for w in class_weights]}")
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=50,
        save_total_limit=3,
        per_device_train_batch_size=16,  # Retained for efficiency
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # Effective batch 32
        num_train_epochs=5,  # Retained for better convergence
        learning_rate=5e-5,  # Increased for improved optimization
        weight_decay=0.01,  # Retained
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to="none",
        warmup_ratio=0.15,  # Retained
        lr_scheduler_type="linear",
        dataloader_drop_last=True,
        fp16=torch.cuda.is_available(),  # Enable mixed precision
        remove_unused_columns=True,
        seed=42,
        optim="adamw_torch",
    )
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.015  # Retained
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    trainer = WeightedTrainer(
        class_weights=class_weights,
        processing_class=tokenizer,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )
    return trainer, training_args

def evaluate_and_save(trainer, test_dataset, df, label_encoder, num_labels, train_df, val_df, test_df):
    """Evaluate, visualize, and save artifacts including setup_training function."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"./model/email_classifier_{timestamp}"
    results_dir = f"./results_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    predictions = trainer.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_true = predictions.label_ids
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))
    final_metrics = compute_metrics((predictions.predictions, y_true))
    print(f"\n{'='*60}\nFINAL PERFORMANCE\n{'='*60}")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Macro F1: {final_metrics['f1_macro']:.4f}")
    print(f"Weighted F1: {final_metrics['f1_weighted']:.4f}")
    print(f"Micro F1: {final_metrics['f1_micro']:.4f}")
    print(f"Train/Val/Test: {len(train_df):,}/{len(val_df):,}/{len(test_df):,}")
    print(f"Classes: {num_labels}")
    create_visualizations(df, label_encoder, y_true, y_pred, results_dir)
    trainer.save_model(model_dir)
    trainer.processing_class.save_pretrained(model_dir)  # Use processing_class to save tokenizer
    with open(f'{model_dir}/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    # Get setup_training function source code
    import inspect
    setup_training_source = inspect.getsource(setup_training)
    # Extract training_args and class_weights from trainer
    training_args_dict = trainer.args.to_dict()
    class_weights = trainer.class_weights.cpu().numpy().tolist() if trainer.class_weights is not None else None
    model_metadata = {
        "model_name": "distilbert-base-uncased",
        "num_labels": num_labels,
        "labels": label_encoder.classes_.tolist(),
        "max_length": 512,
        "timestamp": timestamp,
        "performance": final_metrics,
        "dataset_info": {
            "total_samples": len(df),
            "training_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "class_distribution": df['label'].value_counts().to_dict()
        },
        "training_parameters": training_args_dict,
        "class_weights": class_weights,
        "setup_training_function": setup_training_source  # Save the function source code
    }
    with open(f'{model_dir}/model_metadata.json', 'w') as f:
        json.dump(model_metadata, f, indent=2)
    with open(f'{results_dir}/classification_report.txt', 'w') as f:
        f.write(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4, zero_division=0))
    print(f"\n✅ Training complete! Model: {model_dir}, Results: {results_dir}")
    print("\nSupported categories:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {i}: {label} ({len(df[df['label'] == label]):,} samples)")

def main():
    train_df, val_df, test_df, label_encoder, num_labels = prepare_data()
    train_dataset, val_dataset, test_dataset, tokenizer, model = setup_datasets_and_model(train_df, val_df, test_df, num_labels)
    trainer, training_args = setup_training(train_dataset, val_dataset, train_df, model, tokenizer, num_labels)
    print(f"\n🚀 Training {num_labels}-class classifier (effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps})")
    trainer.train()
    evaluate_and_save(trainer, test_dataset, pd.concat([train_df, val_df, test_df]), label_encoder, num_labels, train_df, val_df, test_df)

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()