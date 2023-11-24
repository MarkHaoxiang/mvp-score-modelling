import argparse
import copy
import logging

from datasets import load_dataset, Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from mvp_score_modelling.nn import ClassificationNet
from mvp_score_modelling.utils import process_from_raw, compute_accuracy

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 2

parser = argparse.ArgumentParser(
    prog="train_gender_classifier",
    description="Trains a gender classifier on the CelabA-HQ dataset"
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=True,
    help="Checkpoint file to output"
)
parser.add_argument(
    "-lr",
    "--learning_rate",
    type=float,
    default=1e-3,
    help="Optimiser learning rate"
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=15,
    help="Number of training epochs"
)

args = parser.parse_args()

def main(output_file: str,
         batch_size: int = 32,
         n_epochs: int = 15,
         train_data_percentage: float = 0.4,
         lr=1e-3):
    # Dataset
    dataset: Dataset = load_dataset("Ryan-sjtu/celebahq-caption")
        # Preprocessing
    dataset: Dataset = dataset['train'].with_format("torch", device=DEVICE)
    dataset = dataset.map(lambda x: {
        'label': 
            torch.tensor([
                1 if 'woman' in x['text'] else 0,
                0 if 'woman' in x['text'] else 1
            ],
            dtype=torch.float32,
            device=DEVICE)
    })
        # Train test split
    dataset = dataset.train_test_split(test_size=0.05)
        # Dataloaders
    train_dataloader = DataLoader(dataset["train"], batch_size=batch_size)
    test_dataloader =  DataLoader(dataset["test"], batch_size=batch_size)
        # Class weightings
    weights = torch.zeros((N_CLASSES,)).to(device=DEVICE)
    n_batches = 0
    for _, data in enumerate(train_dataloader):
        n_batches += 1
        labels = data['label']
        weights += torch.sum(labels, dim=0)
    weights = (sum(weights) / N_CLASSES) / weights
    n_train_batches = int(train_data_percentage * n_batches)

    # Model
    model = ClassificationNet(n_classes=N_CLASSES).to(device=DEVICE)
    best_model = copy.deepcopy(model)

    # Wandb
    wandb.init(
        project="mvp",
        config={
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "lr": lr
        },
        name="gender_classifier",
        mode="online"
    )
    wandb.watch(model, log="all", log_freq=n_batches)

    # Training Loop
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_accuracy = 0
    best_model = model
    with tqdm(total=n_epochs) as pbar:
        train_loss = 0
        for epoch in range(n_epochs):
            for i, data in enumerate(train_dataloader):
                if i > n_train_batches:
                    break
                X, y = data['image'], data['label']
                X = process_from_raw(X)

                optim.zero_grad()
                y_p = model(X)
                loss = loss_fn(y_p, y)
                train_loss += loss.item()
                loss.backward()
                optim.step()
            train_loss = train_loss / n_batches

            # Logging
            accuracy, loss = compute_accuracy(test_dataloader, model, loss_fn, process_from_raw)
            if accuracy >= best_accuracy:
                logging.info(f"Model Improved {epoch} {accuracy}")
                best_accuracy = accuracy
                best_model.load_state_dict(model.state_dict())
            pbar.set_description(f"test accuracy {accuracy} loss {loss}")
            pbar.update(1)

            wandb.log({
                "train/loss": train_loss,
                "evaluation/accuracy": accuracy,
                "evaluation/loss": loss
            })
    
    # Save model
    wandb.unwatch()
    wandb.finish()
    torch.save(best_model, output_file)

if __name__ == "__main__":
    main(
        args.output,
        lr=args.learning_rate,
        n_epochs=args.n_epochs
    )
