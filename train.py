# load a dataset (does it have to be images or can it be just any random image ?)
# for each image train only one linear layer to map one to the other

# Sooo thoughts here. We train on the other network's answers on not on the ground truth (a bit like distillation ?)
# Does this have consequences ?
# Also : they use the dataset that was used to train the original networks. Does it make sense to use another one ? (here cifar because easiest)

from data import get_dataloader
from archs import initialize_vision_module
import torch
from torch import nn
from torch.nn import Sequential
import torch.optim as optim

# PARAMS
output_path = "/mnt/efs/fs1/logs/vitfeatures"
device = "cuda"


def train_model(model, train_data_loader, n_epochs=10, label_model=None, verbose=False):
    """
    if label_model is empty, then training occurs normaly
    otherwise, training will use the labels given by the label_model for a given input
    """
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _labels = data
            labels = _labels if label_model is None else label_model(inputs)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(
                torch.Tensor(inputs)
            )  # Mat 24/02/2022 stopped here to solve AttributeError: 'list' object has no attribute 'shape'
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if verbose and i % 100 == 99:  # print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    return model


# if __name__ == "__main__":
#     pass
