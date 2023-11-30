import os
import torch
import shutil
from datasets import load_dataset, concatenate_datasets, DatasetDict
import argparse

def main(train_percent, test_percent, save_path, num_proc):
    DATASET_SOURCE = "Ryan-sjtu/celebahq-caption"

    dataset = load_dataset(DATASET_SOURCE)

    def label_dataset(batch):
        # `batch` is a dictionary where each value is a list of entries
        texts = batch['text']
        batch['label'] = [1 if 'woman' in text.lower() else 0 for text in texts]
        return batch

    ### Preprocessing
    ds = dataset['train'].map(label_dataset, batched=True, batch_size=32, remove_columns=['text'], num_proc=num_proc)

    dataset_0 = ds.filter(lambda batch: [x==0 for x in batch['label']], batched=True, batch_size=32, num_proc=num_proc)
    dataset_1 = ds.filter(lambda batch: [x==1 for x in batch['label']], batched=True, batch_size=32, num_proc=num_proc)
    print(f"Dataset 0 (male) size: {len(dataset_0)}, Dataset 1 (female) size: {len(dataset_1)}")

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
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    # test even distribution of labels
    print(f"Train dataset label distribution: {torch.unique(torch.tensor(train_dataset['label']), return_counts=True)}")
    print(f"Test dataset label distribution: {torch.unique(torch.tensor(test_dataset['label']), return_counts=True)}")

    ### Save images
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
        print(f"Deleted {save_path}")
    os.makedirs(save_path, exist_ok=True)

    def create_save_image_function(save_directory):
        # create closure to encapsulate save_directory
        def save_image(example, index):
            import os
            image = example['image']
            filepath = os.path.join(save_directory, f'image_{index}.png')
            image.save(filepath)
            return example  # Return the unmodified example
        return save_image

    save_image_function = create_save_image_function(save_path)
    test_dataset.map(save_image_function, with_indices=True, batched=False, num_proc=num_proc)
    print(f"Saved {len(test_dataset)} images to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and save images.")
    parser.add_argument("--train_percent", type=float, default=0.45, required=True, help="Fraction of the dataset to use for training.")
    parser.add_argument("--test_percent", type=float, default=0.05, required=True, help="Fraction of the dataset to use for testing.")
    parser.add_argument("--save_path", type=str, default="output/celebahq/test/all", required=True, help="Path to save the images.")
    parser.add_argument("--num_proc", type=int, default=6, help="Number of processes for parallel execution.")

    args = parser.parse_args()
    main(args.train_percent, args.test_percent, args.save_path, args.num_proc)

# use case:
"""sh
python mvp_score_modelling/evaluation/save_celeb.py \
    --train_percent 0.45 \
    --test_percent 0.05 \
    --save_path output/celebahq/test/all \
    --num_proc 6
"""

"""windows
python mvp_score_modelling/evaluation/save_celeb.py ^
    --train_percent 0.45 ^
    --test_percent 0.05 ^
    --save_path output/celebahq/test/all ^
    --num_proc 6
"""