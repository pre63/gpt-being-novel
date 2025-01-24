import torch
import torch.nn as nn


class PENN(nn.Module):
  def __init__(self, input_size, entanglement_pairs):
    super(PENN, self).__init__()
    self.entanglements = nn.Parameter(torch.randn(entanglement_pairs, input_size))
    self.entanglement_strength = nn.Parameter(torch.ones(entanglement_pairs))

    # Adjust fc1 input size to match combined features (original input + entangled features)
    combined_feature_size = input_size + entanglement_pairs
    self.fc1 = nn.Linear(combined_feature_size, 128)
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    # Compute entangled features
    entangled = torch.einsum("ij,bj->bi", self.entanglements, x)
    entangled = entangled * self.entanglement_strength.view(1, -1)

    # Combine original features with entangled features
    combined_features = torch.cat([x, entangled], dim=1)

    # Forward pass through fully connected layers
    x = torch.relu(self.fc1(combined_features))
    output = self.fc2(x)
    return output
