import torch
import torch.nn as nn


class SAMNN(nn.Module):
  def __init__(self, input_dim, output_dim, initial_modules=2):
    super(SAMNN, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.module_list = nn.ModuleList()  # Properly initialize as ModuleList
    self._initialize_modules(initial_modules)

  def _initialize_modules(self, initial_modules):
    for _ in range(initial_modules):
      module = nn.Sequential(nn.Linear(self.input_dim, 32), nn.ReLU(), nn.Linear(32, self.output_dim))
      self.module_list.append(module)  # Add to ModuleList

  def forward(self, x):
    outputs = torch.stack([module(x) for module in self.module_list], dim=0)
    return torch.mean(outputs, dim=0)
