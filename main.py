import numpy as np
from model import MyModel
from data import *

EPOCHS = 50
BATCH_SIZE = 128
learning_rate = 0.0001

train_loader, test_loader = get_mnist()

architecture = [{"layer_size": 256, "activation": "sigmoid"},  # w: (784, 1024)
                {"layer_size": 10, "activation": "softmax"}]   # w: (1024, 10)

model = MyModel(architecture, 28*28, "cross_entropy_softmax", learning_rate)

history = []
for i in range(EPOCHS):
    print(f"Epochs {i+1}/{EPOCHS}")
    print('------------------------')

    total_loss = 0.0
    total_acc = 0.0

    for step, (imgs, label) in enumerate(train_loader):
        imgs = imgs.reshape(BATCH_SIZE, -1).detach().numpy()
        labels = label.detach().numpy()
        if imgs.shape != (128, 784):
            break
        out = model.forward(imgs)
        loss = model.calculate_loss(out, labels)

        total_loss += np.mean(loss)
        model.backward()
        model.optimize()

    steps_per_epoch = 60000 // BATCH_SIZE
    average_loss = total_loss / steps_per_epoch

    history.append(average_loss)

    print(f'Loss: {round(average_loss, 4)}')
    # print(f'Accuracy: {average_accuracy:.4f}')
    print('------------------------')
print('Training completed.')
