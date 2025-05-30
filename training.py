import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import pickle
from stem import create_training_data
from nnModel import NeuralNet

def train_model():
    # Tạo training data
    train_x, train_y, words, classes = create_training_data()

    # Lưu words và classes
    with open('words.pkl', 'wb') as f:
        pickle.dump(words, f)
    with open('classes.pkl', 'wb') as f:
        pickle.dump(classes, f)

    # Chuyển đổi data thành tensor
    X = torch.FloatTensor(train_x)
    y = torch.FloatTensor(train_y)

    # Khởi tạo model
    input_size = len(X[0])
    hidden_size = 8
    num_classes = len(classes)
    model = NeuralNet(input_size, hidden_size, num_classes)

    # Loss và optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        # Forward pass
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward và optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Final loss: {loss.item():.4f}')

    # Lưu model
    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "words": words,
        "classes": classes
    }
    torch.save(data, "data.pth")
    print("Training completed. Model saved to 'data.pth'")

if __name__ == "__main__":
    train_model() 