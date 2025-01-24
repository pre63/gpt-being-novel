import torch
from samnn_model import SAMNN

# Instantiate the model
input_dim = 10
output_dim = 1
samnn = SAMNN(input_dim, output_dim)

# Generate test data
x_train = torch.randn(100, input_dim)
y_train = torch.sum(x_train, dim=1, keepdim=True)

# Train the model
optimizer = torch.optim.Adam(samnn.parameters(), lr=0.001)
epochs = 500  # Increased epochs for better training

for epoch in range(epochs):
  optimizer.zero_grad()
  output = samnn.forward(x_train)
  loss = torch.nn.MSELoss()(output, y_train)
  loss.backward()
  optimizer.step()
  if epoch % 50 == 0:  # Debug loss every 50 epochs
    print(f"Epoch {epoch}: Loss = {loss.item()}")

# Evaluate
output = samnn.forward(x_train).detach()
mse = torch.mean((output - y_train) ** 2).item()
print("SAMNN MSE:", mse)
print("Test Passed:", mse < 1.0)
