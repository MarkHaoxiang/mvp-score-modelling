import argparse
import copy
import logging

from datasets import load_dataset, Dataset, concatenate_datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from diffusers import ScoreSdeVeScheduler, ScoreSdeVePipeline

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
parser.add_argument(
    "--rff",
    type=int,
    default=64,
    help="Number of random fourier features"
)
parser.add_argument(
    "--noise-batch",
    type=int,
    default=64,
    help="noise batch size"
)
parser.add_argument(
    "--noise-max",
    type=int,
    default=100,
    help="max timestep for random noise, the higher it is, the higher the max noise"
)

args = parser.parse_args()

def get_timesteps_and_sigmas(scheduler, batch_size, max_timesteps=None):
    timesteps = torch.randint(
                    low=0,
                    high=len(scheduler.discrete_sigmas) - 1 \
                        if max_timesteps is None else max_timesteps,
                    size= (batch_size, )
                ).to(device=DEVICE)
    sigmas = scheduler.discrete_sigmas.to(DEVICE)[timesteps]
    return timesteps, sigmas

def main(output_file: str,
         rff: int = 64,   
         noise_batch: int = 64,
         noise_max: int = 100,
         batch_size: int = 32,
         n_epochs: int = 15,
         train_data_percentage: float = 0.4,
         lr=1e-3):
    # Dataset
    dataset: Dataset = load_dataset("Ryan-sjtu/celebahq-caption")

    train_percent = 0.005
    test_percent = 0.001

    train_size = int(len(dataset["train"]) * train_percent)
    test_size = int(len(dataset["train"]) * test_percent)
    print(f"Train size: {train_size}, Test size: {test_size}")

    def label_dataset(batch):
        # `batch` is a dictionary where each value is a list of entries
        texts = batch['text']
        # Apply the label logic to each entry in the batch
        batch['label'] = [1 if 'woman' in text.lower() else 0 for text in texts]
        return batch

    ds = dataset['train'].map(label_dataset, batched=True, batch_size=32, remove_columns=['text'], num_proc=4)

    dataset_0 = ds.filter(lambda batch: [x==0 for x in batch['label']], batched=True, batch_size=32, num_proc=4)
    dataset_1 = ds.filter(lambda batch: [x==1 for x in batch['label']], batched=True, batch_size=32, num_proc=4)
    print(f"Dataset 0 size: {len(dataset_0)}, Dataset 1 size: {len(dataset_1)}")

    def split_dataset(dataset, train_size, test_size):
        train = dataset.select(range(train_size))
        test = dataset.select(range(train_size, train_size + test_size))
        return train, test

    train_size_half = int(len(ds) * train_percent) // 2
    test_size_half = int(len(ds) * test_percent) // 2
    train_dataset_0, test_dataset_0 = split_dataset(dataset_0, train_size_half, test_size_half)
    train_dataset_1, test_dataset_1 = split_dataset(dataset_1, train_size_half, test_size_half)

    # Combine 
    train_dataset = concatenate_datasets([train_dataset_0, train_dataset_1])
    test_dataset = concatenate_datasets([test_dataset_0, test_dataset_1])

    train_dataset = train_dataset.shuffle(seed=0)
    test_dataset = test_dataset.shuffle(seed=0)

    # To tensor
    train_dataset.set_format(type='torch', columns=['image', 'label'])
    test_dataset.set_format(type='torch', columns=['image', 'label'])

    # Process (use process_from_raw on images)
    def get_process_fn(process_from_raw):
        def process(batch):
            batch['image'] = [process_from_raw(img) for img in batch['image']]
            return batch
        return process

    process = get_process_fn(process_from_raw)
    train_dataset = train_dataset.map(process, batched=True, batch_size=5, num_proc=4)
    test_dataset = test_dataset.map(process, batched=True, batch_size=5, num_proc=4)

    # test even distribution of labels
    print(f"Train dataset label distribution: {torch.unique(train_dataset['label'], return_counts=True)}")
    print(f"Test dataset label distribution: {torch.unique(test_dataset['label'], return_counts=True)}")

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader =  DataLoader(test_dataset, batch_size=batch_size)
    print(f"Train size: {len(train_dataloader)}, Test size: {len(test_dataloader)}")

    # Model
    model = ClassificationNet(n_classes=N_CLASSES, rff_dim=rff).to(device=DEVICE)
    best_model = copy.deepcopy(model)

    # Schedueler for getting timesteps and sigmas
    PRETRAINED = "google/ncsnpp-celebahq-256"
    unconditional_pipeline = ScoreSdeVePipeline.from_pretrained(PRETRAINED).to(device=DEVICE)
    scheduler: ScoreSdeVeScheduler = unconditional_pipeline.scheduler

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
    n_batches = len(train_dataloader)
    wandb.watch(model, log="all", log_freq=n_batches)

    # Training Loop
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0
    best_model = model
    with tqdm(total=n_epochs) as pbar:
        train_loss = 0
        for epoch in range(n_epochs):
            for i, data in enumerate(train_dataloader):
                X, y = data['image'], data['label']

                ### Preprocess image
                X = X.unsqueeze(1)
                timesteps, sigmas = get_timesteps_and_sigmas(scheduler, noise_batch, noise_max)
                sigmas = sigmas.repeat(X.shape[0])
                X = scheduler.add_noise(
                    original_samples=X,
                    noise=None,
                    timesteps=timesteps
                ).flatten(0,1)
                y = y.unsqueeze(-1)
                y = y.expand(y.shape[0],noise_batch).flatten() 

                optim.zero_grad()
                y_p = model(X, sigmas)
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
            pbar.set_description(f"test accuracy {accuracy:.4f} loss {loss:.4f}")
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
        rff=args.rff,
        lr=args.learning_rate,
        n_epochs=args.n_epochs
    )
