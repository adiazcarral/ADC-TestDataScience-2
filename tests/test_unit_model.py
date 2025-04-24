# tests/test_unit_model.py
import torch
from src.adc_testdatascience_1.models.logistic import LogisticRegression


def test_logistic_regression_output_shape():
    model = LogisticRegression()
    input_tensor = torch.randn(8, 1, 28, 28)  # Batch size 8
    output = model(input_tensor)
    assert output.shape == (8, 10), f"Expected output shape (8, 10), got {output.shape}"