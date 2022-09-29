import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # self.i2o = nn.Linear(input_size + hidden_size, output_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # combined = torch.cat((input, hidden), 1)
        # hidden = self.i2h(combined)
        # output = self.i2o(combined)
        # if torch.isnan(hidden).any():
        #     print("hidden", hidden)
        # if torch.isnan(output).any():
        #     print("output", output)
        # output = self.softmax(output)
        output, hidden = self.gru(input, hidden)
        output = self.out(output[0])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


# torch.autograd.set_detect_anomaly(True)
