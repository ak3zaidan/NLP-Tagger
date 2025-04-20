from torch import nn
import torch

##################################
#  Q2
##################################

class ParityLSTM(nn.Module) :

    def __init__(self, hidden_dim=16):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=1, 
                           hidden_size=hidden_dim, 
                           num_layers=1, 
                           batch_first=True)
        
        # lin layer to map hidden state to 2 outputs
        self.linear = nn.Linear(hidden_dim, 2)

    def forward(self, x, x_lens):
        outputs, _ = self.lstm(x)

        batch_size = x.size(0)
        
        indices = torch.arange(batch_size, device=x.device)
        
        last_indices = x_lens - 1
        
        last_hidden = outputs[indices, last_indices]
        output = self.linear(last_hidden)
        
        return output
