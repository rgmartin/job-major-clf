import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


install("torch==2.2.*")
install("transformers[torch]==4.37.*")
install("datasets==2.16.*")
install("scikit-learn==1.4.*")

import argparse
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, EarlyStoppingCallback,
                          IntervalStrategy, Trainer, TrainingArguments)

logging.basicConfig(format="%(asctime)s %(funcName)s %(message)s")
logging.getLogger().setLevel(logging.INFO)
logging.info("Setting up log level and format")


def _parse_args():
    parser = argparse.ArgumentParser()
    # Data, model, and output directories
    parser.add_argument(
        "--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"]
    )
    parser.add_argument("--test-dir", type=str, default=os.environ["SM_CHANNEL_TEST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument(
        "--epochs",
        type=int,
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
    )
    parser.add_argument(
        "--base-model-name",
        type=str,
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
    )

    return parser.parse_known_args()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


if __name__ == "__main__":
    # Process arguments
    args, _ = _parse_args()

    model_ckpt = args.base_model_name

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    # Tokenize dataset
    ds = load_dataset(
        "parquet",
        data_files={
            "train": [str(p) for p in Path(args.training_dir).glob("**/*.parquet")],
            "test": [str(p) for p in Path(args.test_dir).glob("**/*.parquet")],
        },
    )

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    ds_encoded = ds.map(tokenize, batched=True, batch_size=None)

    # Config variables
    num_labels = len(ds["train"].to_pandas().label.unique())
    id2label = {id: l for id, l in enumerate(ds["train"].features["label"].names)}
    label2id = {l: id for id, l in id2label.items()}

    config = AutoConfig.from_pretrained(
        model_ckpt, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_ckpt, config=config
    ).to(device)

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    batch_size = args.train_batch_size
    logging_steps = len(ds["train"]) // batch_size
    output_dir = args.model_dir
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy=IntervalStrategy.EPOCH,
        disable_tqdm=False,
        logging_steps=50,
        logging_strategy=IntervalStrategy.STEPS,
        load_best_model_at_end=True,
        save_strategy=IntervalStrategy.EPOCH,
        # save_total_limit=1,
        push_to_hub=False,
        log_level="error",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=ds_encoded["train"],
        eval_dataset=ds_encoded["test"],
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        ],
    )
    trainer.train()
    trainer.save_model()
