import torch
from penn_model import PENN
from sklearn.metrics import mean_squared_error

# Instantiate the model
input_size = 10
entanglement_pairs = 3
penn = PENN(input_size, entanglement_pairs)

# Generate test data
x_train = torch.randn(100, input_size)
y_train = torch.sum(x_train, dim=1, keepdim=True)

# Train the model
optimizer = torch.optim.Adam(penn.parameters(), lr=0.001)
epochs = 500  # Increased epochs for better training

for epoch in range(epochs):
  optimizer.zero_grad()
  output = penn(x_train)
  loss = torch.nn.MSELoss()(output, y_train)
  loss.backward()
  optimizer.step()
  if epoch % 50 == 0:  # Debug loss every 50 epochs
    print(f"Epoch {epoch}: Loss = {loss.item()}")

# Evaluate
output = penn(x_train).detach().numpy()
mse = mean_squared_error(y_train.numpy(), output)
print("PENN MSE:", mse)
print("Test Passed:", mse < 1.0)
