import os
import torch
import numpy as np
import argparse
from src.adc_testdatascience_1.models.logistic import LogisticRegression
from src.adc_testdatascience_1.models.cnn import SimpleCNN
from src.adc_testdatascience_1.models.equivariant_cnn import RotEquivariantCNN
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders
from src.adc_testdatascience_1.scripts.test_model import ModelEvaluator

def test_full_pipeline(model_name="logistic"):
    _, _, test_loader = get_dataloaders(subset_fraction=1.0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = {
        "logistic": ("LogisticRegression", LogisticRegression(), os.path.join("src", "adc_testdatascience_1", "models", "logistic.pth")),
        "cnn": ("SimpleCNN", SimpleCNN(), os.path.join("src", "adc_testdatascience_1", "models", "cnn.pth")),
        "rotcnn": ("RotEquivariantCNN", RotEquivariantCNN(), os.path.join("src", "adc_testdatascience_1", "models", "rotcnn.pth")),
    }

    if model_name not in model_paths:
        raise ValueError(f"Unknown model name: {model_name}")

    model_label, model, model_path = model_paths[model_name]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    evaluator = ModelEvaluator(device=device)
    evaluator.evaluate(model, test_loader, name=model_name)

    y_true = np.array(evaluator.results[model_name]["true_labels"])
    y_pred = np.array(evaluator.results[model_name]["pred_labels"])

    assert isinstance(y_true, np.ndarray), f"Expected numpy.ndarray, got {type(y_true)}"
    assert isinstance(y_pred, np.ndarray), f"Expected numpy.ndarray, got {type(y_pred)}"
    assert len(y_true) == len(y_pred), f"Length mismatch: {len(y_true)} vs {len(y_pred)}"

    evaluator.plot_confusion_matrix(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["logistic", "cnn", "rotcnn"], default="logistic"
    )
    args = parser.parse_args()

    test_full_pipeline(model_name=args.model)
