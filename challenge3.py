from transformers import AutoTokenizer, TrainingArguments, Trainer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoModel

import random
SEED = 42
random.seed(SEED)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')



# Do not change function signature
def preprocess_function(examples: dict) -> dict:
    """
    Tokenize the input texts.

    Input:
        examples: {
            "text": List[str],
            "label": List[int]
        }

    Output:
        {
            "input_ids": Tensor [B, L],
            "attention_mask": Tensor [B, L]
        }
    """
    tokenized_sample = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=400,
    )
    return tokenized_sample


class LSTMSequenceClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 738,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_labels: int = 2,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_token_id,
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask=None, labels=None):
        embedded = self.embedding(input_ids)
        outputs, _ = self.lstm(embedded)

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).clamp(min=1)
            last_index = lengths - 1
            pooled = outputs[torch.arange(outputs.size(0), device=outputs.device), last_index]
        else:
            pooled = outputs[:, -1, :]

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


# Do not change function signature
def init_model() -> nn.Module:
    """
    Initialize a sequence classification model.

    Input:
        input_ids: [B, L]

    Output:
        logits: [B, 2]
    """
    model = LSTMSequenceClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=768,
        hidden_size=256,
        num_layers=2,
        num_labels=2,
        pad_token_id=tokenizer.pad_token_id,
    )

    pretrained_distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
    model.embedding.weight.data.copy_(
        pretrained_distilbert.embeddings.word_embeddings.weight.data
    )
    model.embedding.weight.requires_grad = False
    return model


# Do not change function signature
def train_model(
    model: nn.Module,
    imdb_dataset: Dataset,
    amazon_dataset: Dataset
) -> nn.Module:
    """
    Train the model using:

    - imdb_dataset: large source dataset (movie reviews)
    - amazon_dataset: small target dataset (product reviews)

    Each dataset sample contains:
        input_ids: [L]
        attention_mask: [L]
        labels: int (0 or 1)

    Goal:
        Train a model that generalizes well to the Amazon domain.

    You are free to:
        - train on one or both datasets
        - mix datasets
        - fine-tune sequentially
        - use Trainer or custom loops

    Return:
        trained model
    """
    training_args = TrainingArguments(
        output_dir="data_scratch/results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=128,
        num_train_epochs=10,
        weight_decay=0.05,
        eval_strategy='epoch',
        report_to=[],  # Disable wandb and other external loggers if needed
        save_strategy="no",
        logging_strategy="epoch",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=imdb_dataset,
        eval_dataset=amazon_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer.model


# ===========================================
## HELPERS
# ===========================================
def evaluate_model(model: nn.Module, amazon_test):
    training_args = TrainingArguments(
        output_dir="data_scratch/results_eval",
        per_device_eval_batch_size=128,
        report_to=[],
        do_train=False,
        do_eval=False,
    )

    trainer = Trainer(model=model, args=training_args)
    pred = trainer.predict(amazon_test)
    predictions = np.argmax(pred.predictions, axis=-1)
    labels = pred.label_ids
    accuracy = np.mean(predictions == labels)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    return {"accuracy": float(accuracy)}


def get_data():
    imdb = load_dataset("imdb")
    tokenized_imdb = imdb.map(preprocess_function, batched=True)



    imdb_train = tokenized_imdb['train']
    imdb_test = tokenized_imdb['test']

    # amazon = load_dataset('amazon_polarity')
    # amazon_train = amazon['train']
    # amazon_test = amazon['test']
    
    # print( amazon_train[0])
    # print( imdb_train[0])

    return imdb_train, imdb_train, imdb_test


def run():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    # Get datasets for training and testing
    imdb_train, amazon_train, amazon_test = get_data()

    # Initialize the model using student's init_model function
    model = init_model()

    # Train the model using student's train_model function
    model = train_model(model, imdb_train, amazon_train)

    # Evaluate the model on the test set
    evaluate_model(model, amazon_test)

    return 0


if __name__ == "__main__":
    run()
