from torch import nn

import torch
import torch.nn as nn
import torch.nn.functional as F



class DDRNeuralNetwork(nn.Module):
    def __init__(self, sw, n):
        super(DDRNeuralNetwork, self).__init__()

        # For the list of integers (assuming these are categorical features)
        self.embedding = nn.Embedding(num_embeddings=100, embedding_dim=10)  # Adjust num_embeddings as needed

        # For the 2D array of floats
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1_sw = nn.Linear(32 * (sw // 2) * (sw // 2), 128)  # Adjust size according to your conv/pool layers output

        # For the scalar inputs
        self.fc1_scalar = nn.Linear(2, 64)  # Assuming there are two scalar inputs

        # Final fully connected layers
        self.fc2 = nn.Linear(128 + 64 + 10 * n, 256)  # 10 * n for embeddings output, adjust dimensions
        self.fc3 = nn.Linear(256, n)  # Output the same size as the target

    def forward(self, prev_measure, spectrogram_data, difficulty_and_bpm):
        # Integer inputs handling
        x_integers = self.embedding(prev_measure)

        # 2D array inputs handling
        x_sw = spectrogram_data.unsqueeze(1)  # Adding a channel dimension
        x_sw = F.relu(self.conv1(x_sw))
        x_sw = self.pool(x_sw)
        x_sw = F.relu(self.conv2(x_sw))
        x_sw = x_sw.view(x_sw.size(0), -1)
        x_sw = F.relu(self.fc1_sw(x_sw))

        # Scalar inputs handling
        x_scalars = F.relu(self.fc1_scalar(difficulty_and_bpm))

        # Concatenate all features
        x_combined = torch.cat((x_integers.view(x_integers.size(0), -1), x_sw, x_scalars), dim=1)
        
        # Final layers
        x_combined = F.relu(self.fc2(x_combined))
        out = self.fc3(x_combined)
        return out
    
def createNetworkAndGetDevice(sw, n):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DDRNeuralNetwork(sw, n).to(device), device


def trainModelBatch(model, data, optimizer, criterion, device):
    for datum in data:
        optimizer.zero_grad()
        prev_measure, spectrogram_data, diff_and_bpm, target = [d.to(device) for d in datum]
        output = model(prev_measure, spectrogram_data, diff_and_bpm)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

