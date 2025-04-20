# scripts/train_and_validate.py
import argparse
import os 

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

from src.adc_testdatascience_1.models.cnn import SimpleCNN
from src.adc_testdatascience_1.models.equivariant_cnn import RotEquivariantCNN
from src.adc_testdatascience_1.models.logistic import LogisticRegression
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders


def train_model(model, train_loader, val_loader, device, model_name="model", epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_running_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_running_loss += loss.item() * inputs.size(0)

                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_labels.extend(targets.cpu().numpy())

        val_loss = val_running_loss / len(val_loader.dataset)
        acc = accuracy_score(val_labels, val_preds)
        precision = precision_score(
            val_labels, val_preds, average="macro", zero_division=0
        )
        recall = recall_score(val_labels, val_preds, average="macro", zero_division=0)
        f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)

        # Optional: Confusion matrix
        # conf_matrix = confusion_matrix(val_labels, val_preds)

        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        print(f"📈 Epoch {epoch+1}/{epochs}")
        print(f"    🏋️ Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")
        print(
            f"    🧪 Val Loss: {val_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | Prec: {precision:.4f} | Recall: {recall:.4f}"
        )

    # # Always save the final model at the end
    # torch.save(model.state_dict(), f"{src/adc_testdatascience_1/models/{model_name}}.pth")
    # print(f"✅ Model saved as src/adc_testdatascience_1/models/{model_name}.pth")

    # Define the path
    save_path = os.path.join("src", "adc_testdatascience_1", "models", f"{model_name}.pth")

    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved as {save_path}")


    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["logistic", "cnn", "rotcnn"], default="logistic"
    )
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_dataloaders(subset_fraction=0.05)

    if args.model == "logistic":
        model = LogisticRegression()
    elif args.model == "cnn":
        model = SimpleCNN()
    elif args.model == "rotcnn":
        model = RotEquivariantCNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    train_model(model, train_loader, val_loader, device, model_name)
