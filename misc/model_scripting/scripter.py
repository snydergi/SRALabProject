"""Used to convert trained models to TorchScript format for deployment."""

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

# # LSTM BigData Trial 6 Model
# class JointModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=11, hidden_size=50, num_layers=4, batch_first=True, dropout=0.4)
#         self.dense = nn.Linear(50, 128)
#         self.linear = nn.Linear(128, 4)
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.dense(x)
#         x = self.linear(x)
#         return x

# Joint Model (8-to-1) for Stacking Test
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=8, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
class StackedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize models
        self.j1p = JointModel()
        self.j2p = JointModel()
        self.j3p = JointModel()
        self.j4p = JointModel()
        self.j1v = JointModel()
        self.j2v = JointModel()
        self.j3v = JointModel()
        self.j4v = JointModel()

        # Load pretrained weights
        self.j1p.load_state_dict(torch.load('../../lstm_FullData/trial10/jt1_pos/lstm_model_epoch149.pth'))
        self.j2p.load_state_dict(torch.load('../../lstm_FullData/trial10/jt2_pos/lstm_model_epoch148.pth'))
        self.j3p.load_state_dict(torch.load('../../lstm_FullData/trial10/jt3_pos/lstm_model_epoch147.pth'))
        self.j4p.load_state_dict(torch.load('../../lstm_FullData/trial10/jt4_pos/lstm_model_epoch148.pth'))
        self.j1v.load_state_dict(torch.load('../../lstm_FullData/trial10/jt1_vel/lstm_model_epoch143.pth'))
        self.j2v.load_state_dict(torch.load('../../lstm_FullData/trial10/jt2_vel/lstm_model_epoch149.pth'))
        self.j3v.load_state_dict(torch.load('../../lstm_FullData/trial10/jt3_vel/lstm_model_epoch144.pth'))
        self.j4v.load_state_dict(torch.load('../../lstm_FullData/trial10/jt4_vel/lstm_model_epoch147.pth'))

    def forward(self, x):
        x1p = self.j1p(x)
        x2p = self.j2p(x)
        x3p = self.j3p(x)
        x4p = self.j4p(x)
        x1v = self.j1v(x)
        x2v = self.j2v(x)
        x3v = self.j3v(x)
        x4v = self.j4v(x)

        # Contatenate outputs along the last dimension ([batch, seq_len, 8])
        return torch.cat([x1p, x2p, x3p, x4p, x1v, x2v, x3v, x4v], dim=-1)

# model = JointModel()
# model.load_state_dict(torch.load('../../lstm_FullData/trial6/lstm_model_epoch194.pth'))
# model_scripted = torch.jit.script(model)
# model_scripted.save('scripts/lstm_trial6.pt')

model = StackedModel()
model_sctipted = torch.jit.script(model)
model_sctipted.save('scripts/stacked_future_model.pt')
