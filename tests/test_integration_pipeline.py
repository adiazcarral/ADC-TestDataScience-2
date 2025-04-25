import torch
from torch.utils.data import DataLoader, TensorDataset
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE

def test_training_and_validation():
    # Dummy data (input, target)
    X = torch.randn(8, 500, 26)
    y = torch.randn(8, 100)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=2)
    
    model = LSTMVAE(
        num_input=26,
        num_hidden=128,
        num_layers=2,
        dropout=0.3,
        output_window=1,
        output_dim=1
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = loss_fn(pred, batch_y)
        loss.backward()
        optimizer.step()

    assert isinstance(loss.item(), float)
