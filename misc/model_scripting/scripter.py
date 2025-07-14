import torch
import torch.nn as nn

# # LSTM BigData Trial 4 Model
# class JointModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True, dropout=0.2)
#         self.linear = nn.Linear(50, 4)
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x

# # LSTM BigData Trial 5 Model
# class JointModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=1, batch_first=True)
#         self.linear = nn.Linear(50, 4)
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x

# LSTM BigData Trial 6 Model
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=4, batch_first=True, dropout=0.4)
        self.dense = nn.Linear(50, 128)
        self.linear = nn.Linear(128, 4)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.dense(x)
        x = self.linear(x)
        return x

model = JointModel()
model.load_state_dict(torch.load('../../lstm_BigData/trial6/lstm_model_epoch194.pth'))
model_scripted = torch.jit.script(model)
model_scripted.save('scripts/lstm_trial6.pt')