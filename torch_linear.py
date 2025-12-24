import torch
import torch.nn as nn
import torch.optim as optim

x = torch.tensor([[0.], [1.], [2.], [3.], [4.]])
y = torch.tensor([[1.], [3.], [5.], [7.], [9.]])

model = nn.Linear(1, 1)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(300):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

test_x = torch.tensor([[10.]])
prediction = model(test_x)

print("PyTorch Prediction for x=10:", prediction.item())
