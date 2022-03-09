# load a dataset (does it have to be images or can it be just any random image ?)
# for each image train only one linear layer to map one to the other

# Sooo thoughts here. We train on the other network's answers on not on the ground truth (a bit like distillation ?)
# Does this have consequences ?
# Also : they use the dataset that was used to train the original networks. Does it make sense to use another one ? (here cifar because easiest)

from data import get_dataloader
from archs import initialize_vision_module
import torch

# PARAMS
output_path = "/mnt/efs/fs1/logs/vitfeatures"
device = "cuda"


def train_model(
    model,
    train_data_loader,
    optimiser,
    n_epochs=10,
    label_model=None,
    verbose=False,
    criterion=torch.nn.MSELoss(),
):
    """
    if label_model is empty, then training occurs normaly
    otherwise, training will use the labels given by the label_model for a given input
    """
    model.to(device)
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, _labels = data
            inputs = inputs.to(device)
            labels = _labels if label_model is None else label_model(inputs)
            labels = labels.float().to(device)
            # zero the parameter gradients
            optimiser.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            # print statistics
            # running_loss += loss.item()
            if verbose:  # and i % 100 == 99:  # print every 100 mini-batches
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {loss.item():.3f}"
                )  # running_loss / 2000:.3f}")
                # running_loss = 0.0
    return model


# if __name__ == "__main__":
#     pass
