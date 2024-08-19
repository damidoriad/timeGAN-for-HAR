import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import accuracy_score
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed_input)
        logits = self.fc(h_n[-1])
        y_hat = self.sigmoid(logits)
        return logits, y_hat


def train_test_divide(dataX, dataX_hat, dataT, test_split=0.2):
    # Combine real and fake data
    data = np.concatenate((dataX, dataX_hat), axis=0)
    labels = np.concatenate((np.ones(len(dataX)), np.zeros(len(dataX_hat))), axis=0)
    lengths = np.concatenate((dataT, dataT), axis=0)  # Use same lengths for real and fake
    
    # Convert to tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
    
    # Create dataset
    dataset = TensorDataset(data_tensor, labels_tensor, lengths_tensor)
    
    # Split into train/test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset


def discriminative_score_metrics(dataX, dataX_hat):
    # Parameters
    hidden_dim = max(int(dataX[0].shape[1] / 2), 1)
    iterations = 2000
    batch_size = 128
    learning_rate = 0.001
    
    # Prepare data
    No = len(dataX)
    data_dim = dataX[0].shape[1]
    
    # Compute sequence lengths
    dataT = [len(seq) for seq in dataX]
    
    # Get the maximum sequence length
    Max_Seq_Len = max(dataT)
    
    # Padding the sequences to the maximum length
    dataX_padded = [np.pad(seq, ((0, Max_Seq_Len - len(seq)), (0, 0)), 'constant') for seq in dataX]
    dataX_hat_padded = [np.pad(seq, ((0, Max_Seq_Len - len(seq)), (0, 0)), 'constant') for seq in dataX_hat]
    
    # Divide into training and test sets
    train_dataset, test_dataset = train_test_divide(dataX_padded, dataX_hat_padded, dataT)
    
    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize discriminator
    discriminator = Discriminator(input_dim=data_dim, hidden_dim=hidden_dim)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
    
    # Training
    discriminator.train()
    for epoch in range(iterations):
        for real_data, labels, seq_lengths in train_loader:
            # Forward pass
            logits, predictions = discriminator(real_data, seq_lengths)
            loss = criterion(logits.squeeze(), labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Testing
    discriminator.eval()
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for real_data, labels, seq_lengths in test_loader:
            logits, predictions = discriminator(real_data, seq_lengths)
            y_pred.extend(predictions.squeeze().cpu().numpy())
            y_true.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    accuracy = accuracy_score(y_true, y_pred > 0.5)
    
    # Discriminative score
    disc_score = np.abs(0.5 - accuracy)
    
    return disc_score
