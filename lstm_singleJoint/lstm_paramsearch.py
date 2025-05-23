import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

def load_data():
    """Create datasets for training and testing."""
    # Load the first range (164041 to 367374)
    patient_part1 = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                            skiprows=range(1, 164041), 
                            nrows=367374-164041)
    therapist_part1 = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                                skiprows=range(1, 164041), 
                                nrows=367374-164041)

    # Load the second range (454139 to 654139)
    patient_part2 = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                            skiprows=range(1, 454139), 
                            nrows=654139-454139)
    therapist_part2 = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                                skiprows=range(1, 454139), 
                                nrows=654139-454139)

    # Load the third range (696306 to 894640)
    patient_part3 = pd.read_csv('X2_SRA_A_07-05-2024_10-39-10-mod-sync.csv', 
                            skiprows=range(1, 696306), 
                            nrows=894640-696306)
    therapist_part3 = pd.read_csv('X2_SRA_B_07-05-2024_10-41-46-mod-sync.csv', 
                                skiprows=range(1, 696306), 
                                nrows=894640-696306)

    # Concatenate the two parts
    patient = pd.concat([patient_part1, patient_part2])
    therapist = pd.concat([therapist_part1, therapist_part2])

    patient_data = patient[['JointPositions_1']].values.astype('float32')
    therapist_data = therapist[['JointPositions_3']].values.astype('float32')

    patient_test = patient_part3[['JointPositions_1']].values.astype('float32')
    therapist_test = therapist_part3[['JointPositions_3']].values.astype('float32')

    timeseries = np.column_stack((patient_data, therapist_data))
    timeseries_test = np.column_stack((patient_test, therapist_test))
 
    # plt.plot(timeseries)
    # plt.xlabel('Time Steps (~4ms)')
    # plt.ylabel('Joint Positions')
    # plt.title('Left Hip Joint Positions Over Time')
    # plt.legend(['Patient', 'Therapist'])
    # plt.xlim(0, 7500)
    # plt.show()

    # train-test split for time series
    train_size = int(len(timeseries))
    test_size = int(len(timeseries_test * 0.5))
    train, test = timeseries[:train_size], timeseries_test[:test_size]
    return train.copy(), test.copy(), timeseries.copy(), timeseries_test.copy()

def create_dataset(dataset, lookback):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback, 0]  # Feature is patient data
        target = dataset[i+1:i+lookback+1, 1]  # Target is therapist data
        X.append(feature)
        y.append(target)
    X = np.array(X)
    y = np.array(y)
    X = torch.tensor(X).reshape(-1, lookback, 1)
    y = torch.tensor(y).reshape(-1, lookback, 1)
    return X, y

class JointModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.float()
        x, _ = self.lstm(x)
        # extract only the last time step
        # x = x[:, -1, :]
        x = self.linear(x)
        return x
    
def train_model(train, test, timeseries, timeseries_test):
    lookback = 50  # ~4ms timestep, so ~200ms lookback window
    X_train, y_train = create_dataset(train, lookback=lookback)
    X_test, y_test = create_dataset(test, lookback=lookback)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    model = JointModel()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=256)

    train_loss_list = []
    test_loss_list = []
    n_epochs = 500
    for epoch in range(n_epochs):
        model.train()
        epoch_losses = []
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_loss_list.append(np.mean(epoch_losses))
        # Validation
        # if epoch % 100 != 0:
        #     continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
            test_loss_list.append(loss_fn(y_pred, y_test))
            print(f"Epoch {epoch}: Train RMSE {train_rmse:.4f}, Test RMSE {test_rmse:.4f}")
        torch.save(model.state_dict(), f'lstm_model_epoch{epoch}.pth')
    plot_loss(train_loss_list, test_loss_list)
    plot_predictions(model, X_test, test, lookback, timeseries, timeseries_test)
    return model

def plot_loss(train_loss_list, test_loss_list):
    # Plotting the losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_list, label='Training Loss', alpha=0.8)
    plt.plot(test_loss_list, label='Test Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (RMSE)')
    plt.title('Training and Test Loss Over Epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add a vertical line at the best test loss point
    best_epoch = np.argmin(test_loss_list)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best Epoch ({best_epoch})')
    plt.legend()

    # Save the plot
    plt.savefig('training_loss_curves.png', dpi=300, bbox_inches='tight')

def plot_predictions(model, X_test, test, lookback, timeseries, timeseries_test):
    train_size = int(len(timeseries))
    test_size = int(len(timeseries_test * 0.5))
    # Plotting
    with torch.no_grad():
        # Create arrays for plotting
        therapist_true = np.ones_like(timeseries[:, 1]) * np.nan
        therapist_pred = np.ones_like(timeseries[:, 1]) * np.nan
        
        # Fill in the true therapist data
        therapist_true[lookback:train_size] = timeseries[lookback:train_size, 1]

        # Create arrays for test portion
        test_true = np.ones_like(test[:, 1]) * np.nan
        test_pred = np.ones_like(test[:, 1]) * np.nan
        
        # Fill in true test data
        test_true[lookback:test_size] = test[lookback:test_size, 1]
        
        # Get predictions for test
        test_predictions = model(X_test)[:, -1, :].numpy().flatten()
        test_pred[lookback:test_size] = test_predictions

    # Plot training data
    plt.plot(therapist_true, c='b', label='True Therapist (Train)')
    plt.plot(therapist_pred, c='r', linestyle='--', label='Predicted Therapist (Train)')
    plt.plot(timeseries[:, 0], c='g', alpha=0.3, label='Patient Data (input)')

    # Plot test data (offset by train_size)
    test_x = np.arange(train_size, train_size + test_size)
    plt.plot(test_x, test_true, c='blue', alpha=0.5, label='True Therapist (Test)')
    plt.plot(test_x, test_pred, c='red', linestyle=':', label='Predicted Therapist (Test)')

    plt.xlim(0, train_size + test_size)
    plt.xlabel('Time Steps (~4ms)')
    plt.ylabel('Joint Positions')
    plt.legend()
    plt.title("Therapist Hip Data Prediction from Patient Data")
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig('lstm_therapist_prediction.png', dpi=300, bbox_inches='tight')

def rmse(y_true, y_pred):
    """Calculate RMSE"""
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def main():
    train, test, timeseries, timeseries_test = load_data()
    X_train, y_train = create_dataset(train, lookback=50)
    # train_model(train, test, timeseries, timeseries_test)
    X_train_np = X_train.numpy().reshape(X_train.shape[0], -1)
    y_train_np = y_train.numpy().reshape(y_train.shape[0], -1)
    param_grid = {
        'module__hidden_size': [25, 50, 75, 100],
        'module__num_layers': [1, 2, 3],
        'optimizer__lr': [0.00001, 0.0001, 0.001, 0.01, 0.1],
        'batch_size': [128, 256]
    }
    model = NeuralNetRegressor(
        JointModel,
        max_epochs=10,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        batch_size=256,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        train_split=None
    )
    scorer = make_scorer(rmse, greater_is_better=False)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scorer, n_jobs=1, cv=3, verbose=3, error_score='raise')
    grid_result = grid.fit(X_train, y_train)
    print("Best parameters found: ", grid_result.best_params_)
    print("Best RMSE score: ", -grid_result.best_score_)  # Negative because greater_is_better=False
    return 0

if __name__ == "__main__":
    main()