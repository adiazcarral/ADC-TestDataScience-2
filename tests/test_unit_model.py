import torch
from src.adc_testdatascience_2.models.lstm_vae import LSTMVAE  

def test_lstmvae_forward_pass():
    # Initialize model
    model = LSTMVAE(
        num_input=26,
        num_hidden=128,
        num_layers=2,
        dropout=0.3,
        output_window=1,
        output_dim=1
    )

    sample_input = torch.randn(4, 500, 26)  # batch_size=4, seq_len=500, features=26
    output = model(sample_input)
    
    assert isinstance(output, torch.Tensor)
    assert output.shape[0] == 4  # batch size

