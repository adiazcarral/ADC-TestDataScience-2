import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

from src.adc_testdatascience_1.models.cnn import SimpleCNN
from src.adc_testdatascience_1.models.equivariant_cnn import RotEquivariantCNN
from src.adc_testdatascience_1.models.logistic import LogisticRegression
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders




class ModelEvaluator:
    def __init__(self, device):
        self.device = device
        self.results = {}

    def evaluate(self, model, test_loader, name="Model"):
        model.to(self.device)
        model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        precision = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        self.results[name] = {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "true_labels": all_targets,
            "pred_labels": all_preds,
        }

        print(f"✅ Evaluation - {name}")
        print(f"   Accuracy: {acc:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1 Score: {f1:.4f}")
        unique_classes = np.unique(all_targets)
        print(f"✅ Classes present in test set: {unique_classes}")
        return self.results[name]

    def plot_confusion_matrix(self, model_name):
        y_true = self.results[model_name]["true_labels"]
        y_pred = self.results[model_name]["pred_labels"]
        labels = list(range(10))  # For MNIST: 0 to 9

        cm = confusion_matrix(y_true, y_pred, labels=labels)

        # Normalize rows (true labels) to percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
        cm_percent = np.nan_to_num(cm_percent)  # avoid NaNs for any empty rows

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)

        # Colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Percentage (%)", rotation=-90, va="bottom")

        # Set labels
        ax.set(
            xticks=np.arange(len(labels)),
            yticks=np.arange(len(labels)),
            xticklabels=labels,
            yticklabels=labels,
            xlabel="Predicted label",
            ylabel="True label",
            title=f"Confusion Matrix (%) - {model_name}"
        )

        # Rotate tick labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Annotate each cell with percentage
        for i in range(len(labels)):
            for j in range(len(labels)):
                percentage = cm_percent[i, j]
                ax.text(
                    j, i, f"{percentage:.1f}",
                    ha="center", va="center",
                    color="white" if percentage > 50 else "black"
                )

        plt.tight_layout()
        plt.show()


    def compare_models(self):
        metrics = ["accuracy", "precision", "recall", "f1"]
        model_names = list(self.results.keys())
        scores = {
            metric: [self.results[m][metric] for m in model_names] for metric in metrics
        }

        x = np.arange(len(model_names))
        width = 0.2

        plt.figure(figsize=(10, 6))
        for i, metric in enumerate(metrics):
            plt.bar(x + i * width, scores[metric], width, label=metric)

        plt.xticks(x + width * (len(metrics) / 2 - 0.5), model_names)
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.title("Model Comparison (Accuracy, Precision, Recall, F1)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["logistic", "cnn", "rotcnn"], default="logistic"
    )
    args = parser.parse_args()

    _, _, test_loader = get_dataloaders(subset_fraction=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "logistic":
        model = LogisticRegression()
        model_path = "src/adc_testdatascience_1/models/logistic.pth"
        model_name = "Logistic"
    elif args.model == "cnn":
        model = SimpleCNN()
        model_path = "src/adc_testdatascience_1/models/cnn.pth"
        model_name = "CNN"
    elif args.model == "rotcnn":
        model = RotEquivariantCNN()
        model_path = "src/adc_testdatascience_1/models/rotcnn.pth"
        model_name = "RotEquivariantCNN"

    # Load model
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    evaluator = ModelEvaluator(device=device)
    evaluator.evaluate(model, test_loader, name=model_name)
    evaluator.plot_confusion_matrix(model_name)
