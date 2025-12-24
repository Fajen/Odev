import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)

class TorchQuadraticModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

print("PyTorch Quadratic Model")

x_train = torch.linspace(-10, 10, 200).unsqueeze(1)
y_train = x_train**2 + 2*x_train + 1

x_mean = x_train.mean()
x_std = x_train.std()
x_train_n = (x_train - x_mean) / x_std

model = TorchQuadraticModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 5000
for _ in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train_n)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

test_x = torch.tensor([[-3.], [0.], [2.], [5.]])
test_x_n = (test_x - x_mean) / x_std

model.eval()
with torch.no_grad():
    predictions = model(test_x_n)

for x, y in zip(test_x, predictions):
    print(f"x={x.item():.1f} -> y≈{y.item():.2f}")
