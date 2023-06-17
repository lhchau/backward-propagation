import numpy as np
from model import MyModel
from data import *

EPOCHS = 50
BATCH_SIZE = 128
learning_rate = 0.0001

train_loader, test_loader = get_mnist()

architecture = [{"layer_size": 1024, "activation": "sigmoid"},  # w: (784, 1024)
                {"layer_size": 10, "activation": "softmax"}]   # w: (1024, 10)

model = MyModel(architecture, 28*28, "cross_entropy_softmax", learning_rate)

history = []
for i in range(EPOCHS):
    for imgs, label in train_loader:
        imgs = imgs.reshape(BATCH_SIZE, -1).detach().numpy()
        labels = label.detach().numpy()
        if imgs.shape != (128, 784):
            break
        out = model.forward(imgs)
        loss = model.calculate_loss(out, labels)

        history.append(np.mean(loss))
        model.backward()
        model.optimize()

    print(f"Epochs {i+1}: {np.round(history[-1], 3)}")
