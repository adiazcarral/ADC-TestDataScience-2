import torch
from src.adc_testdatascience_1.models.logistic import LogisticRegression
from src.adc_testdatascience_1.utils.data_utils import get_dataloaders
from src.adc_testdatascience_1.scripts.train_model import train_model

def test_train_model_functional():
    train_loader, val_loader, _ = get_dataloaders(subset_fraction=0.01)
    model = LogisticRegression()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model, history = train_model(model, train_loader, val_loader, device, epochs=1)
    assert len(history["train_loss"]) == 1
    assert isinstance(history["val_acc"][0], float)
