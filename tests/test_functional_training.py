import torch
import os
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE

def test_end_to_end_pipeline():
    model = LSTMVAE(
        num_input=26,
        num_hidden=128,
        num_layers=2,
        dropout=0.3,
        output_window=1,
        output_dim=1
    )
    dummy_input = torch.randn(4, 500, 26)
    dummy_target = torch.randn(4, 100)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    output = model(dummy_input)
    loss = loss_fn(output, dummy_target)
    loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), "test_dummy_model.pth")
    assert os.path.exists("test_dummy_model.pth")
    os.remove("test_dummy_model.pth")
