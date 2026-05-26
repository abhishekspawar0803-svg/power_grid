import torch
import torch.nn as nn


class Autoencoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.encoder = nn.LSTM(self.input_size,self.hidden_size,self.num_layers,batch_first=True)
        self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=32),
            nn.ReLU(),
            nn.Linear(32, self.input_size)
        )
    def forward(self,seq):
        _,(hidden,cell) = self.encoder(seq)
        bottleneck = hidden[-1]
        bottleneck = bottleneck.unsqueeze(1)
        bottleneck = bottleneck.repeat(1,12,1)
        output,(hid,cell2) = self.decoder(bottleneck)
        out = self.output_layer(output)
        return out


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.BatchNorm1d(32),       
            nn.ReLU(),                 
            nn.Dropout(0.2),           
            nn.Linear(32, num_classes) 
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_timestep_out = out[:, -1, :]
        return self.classifier_head(last_timestep_out)