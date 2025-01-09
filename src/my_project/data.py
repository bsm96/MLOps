### To run this file: /opt/anaconda3/bin/python "/Users/bsm/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/DTU/5. semester/02476 Machine Learning Operations/MLOps/src/my_project/data.py" "data/corruptedmnist" "data/processed"

import torch
import typer
import os


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def preprocess_data(raw_dir: str, processed_dir: str) -> None:
    """Process raw data and save it to processed directory."""
    # Ensure the processed directory exists
    os.makedirs(processed_dir, exist_ok=True)

    # Load and concatenate training data
    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(os.path.join(raw_dir, f"train_images_{i}.pt")))
        train_target.append(torch.load(os.path.join(raw_dir, f"train_target_{i}.pt")))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    # Load test data
    test_images: torch.Tensor = torch.load(os.path.join(raw_dir, "test_images.pt"))
    test_target: torch.Tensor = torch.load(os.path.join(raw_dir, "test_target.pt"))

    # Reshape and convert data types
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    # Normalize the images
    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed data
    torch.save(train_images, os.path.join(processed_dir, "train_images.pt"))
    torch.save(train_target, os.path.join(processed_dir, "train_target.pt"))
    torch.save(test_images, os.path.join(processed_dir, "test_images.pt"))
    torch.save(test_target, os.path.join(processed_dir, "test_target.pt"))

    print(f"Data preprocessing complete. Processed data saved in {processed_dir}.")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    # Define paths to processed data
    processed_dir = "data/processed"

    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set


if __name__ == "__main__":
    typer.run(preprocess_data)
