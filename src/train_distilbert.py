import argparse
import os
import pickle
from pathlib import Path

from .activation_logger import ActivationLogger

try:
    from datasets import load_dataset
    from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
except ImportError:  # pragma: no cover - dependencies missing
    load_dataset = None
    DistilBertForSequenceClassification = None
    TrainingArguments = None
    Trainer = None


def train(train_path: Path, val_path: Path, output_dir: Path, epochs: int = 1):
    if load_dataset is None:
        raise ImportError("transformers and datasets packages are required")

    data_files = {"train": str(train_path), "validation": str(val_path)}
    dataset = load_dataset("json", data_files=data_files)

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    logger = ActivationLogger()
    model.distilbert.transformer.register_forward_hook(logger.hook_fn("transformer"))

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset["train"], eval_dataset=dataset["validation"])
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "activations.pkl", "wb") as f:
        pickle.dump(logger.data, f)


def cli():
    p = argparse.ArgumentParser(description="Fine-tune DistilBERT on custom text")
    p.add_argument("train_jsonl", type=Path)
    p.add_argument("val_jsonl", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("model_output"))
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    train(args.train_jsonl, args.val_jsonl, args.output_dir, args.epochs)


if __name__ == "__main__":
    cli()
