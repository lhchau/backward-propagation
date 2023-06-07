from utils import get_mnist
from utils import get_model

EPOCHS = 50
BATCH_SIZE = 128
learning_rate = 0.0001

train_imgs, train_targets, test_imgs, test_targets = get_mnist()

model = get_model()

for i in range(EPOCHS):

    out = model.forward(train_imgs)
    loss = criterion(out, train_targets)
    loss.backward()
