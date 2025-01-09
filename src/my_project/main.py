# To run this code, use the following commands:
# Træn modellen: learning rate, batch size, og antal epochs:
# export PYTHONPATH=$PWD/src
# python src/my_project/main.py train --lr 0.001 --batch-size 64 --epochs 3
# Hvis du vil evaluere en gemt model, kør denne kommando: python src/my_project/main.py evaluate model.pth

import matplotlib.pyplot as plt
import torch
import typer
from my_project.data import corrupt_mnist
from my_project.model import MyAwesomeModel

# Set the device to GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# Create a Typer app
app = typer.Typer()

# Command to train the model
@app.command()
def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 3) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Initialize the model and move it to the device
    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    # Create a DataLoader for the training set
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Dictionary to store training statistics
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            # Store loss and accuracy statistics
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    # Save the trained model
    print("Training complete")
    torch.save(model.state_dict(), "model.pth")

    # Plot and save training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

# Command to evaluate the model
@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    # Load the model and move it to the device
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    # Create a DataLoader for the test set
    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    # Evaluate the model
    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)

    # Print the test accuracy
    print(f"Test accuracy: {correct / total}")

# Run the Typer app
if __name__ == "__main__":
    app()
