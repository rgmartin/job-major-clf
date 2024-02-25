import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('torch==2.2.*')
install('transformers[torch]==4.37.*')
install('datasets==2.16.*')

import json
import os
import pathlib
import tarfile
from pathlib import Path
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoModel, AutoTokenizer, Trainer, AutoModelForSequenceClassification
import logging



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


logging.basicConfig(format='%(asctime)s %(funcName)s %(message)s')
logging.getLogger().setLevel(logging.INFO)
logging.info("Setting up log level and format")

if __name__ == "__main__":


    #All paths are local for the processing container
    model_path = "/opt/ml/processing/model/model.tar.gz"
    test_path = "/opt/ml/processing/test/test.parquet"
    output_dir = "/opt/ml/processing/evaluation"

    # Read model tar file
    try:
        with tarfile.open(model_path, "r:gz") as t:
            logging.info(f'Extracting {model_path} to {os.path.dirname(model_path)} ')
            t.extractall(path=os.path.dirname(model_path))
        model_path = os.path.dirname(model_path)
        logging.info(f'Now model_path = {model_path}')
    except IsADirectoryError:
        logging.info('model_path is a directory')


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = (AutoModelForSequenceClassification
             .from_pretrained(model_path)
             .to(device))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ds = load_dataset("parquet", data_files={ 'test': test_path})
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)
    ds_encoded = ds.map(tokenize, batched=True, batch_size=None)

    trainer = Trainer(model=model, compute_metrics=compute_metrics)
    preds_output = trainer.predict(ds_encoded["test"], )
    print(preds_output.metrics)
    report_dict = {
        "classification_metrics": preds_output.metrics
    }
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/evaluation.json", "w") as f:
        f.write(json.dumps(report_dict))