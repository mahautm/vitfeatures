# load a dataset (does it have to be images or can it be just any random image ?)
# for each image train only one linear layer to map one to the other

# Sooo thoughts here. We train on the other network's answers on not on the ground truth (a bit like distillation ?)
# Does this have consequences ?
# Also : they use the dataset that was used to train the original networks. Does it make sense to use another one ? (here cifar because easiest)

from pyexpat import model
from data import get_dataloader
from archs import initialize_vision_module
import torch
from torch import nn
from torch.nn import Sequential
import torch.optim as optim

# PARAMS
n_epochs = 10
output_path = "/mnt/efs/fs1/logs/vitfeatures"
device = "cuda"


def load_and_pretrain_networks():

    model1, fc_layer1, n_features1 = initialize_vision_module("vit")
    model2, fc_layer2, n_features2 = initialize_vision_module("resnet152")
    return (
        model1.to(device),
        fc_layer1.to(device),
        n_features1.to(device),
        model2.to(device),
        fc_layer2.to(device),
        n_features2.to(device),
    )


def train_linear():
    (
        model1,
        fc_layer1,
        n_features1,
        model2,
        fc_layer2,
        n_features2,
    ) = load_and_pretrain_networks()
    # freeze everything
    for param in model1.parameters():
        param.requires_grad = False
    for param in fc_layer1.parameters():
        param.requires_grad = False
    model1 = model.eval()
    for param in model2.parameters():
        param.requires_grad = False
    for param in fc_layer2.parameters():
        param.requires_grad = False
    model2 = model.eval()

    # split model, and add a linear layer in the midle
    model12 = Sequential(
        model1,
        nn.Linear(n_features1, n_features2),
        fc_layer2,
    )

    model21 = Sequential(
        model2,
        nn.Linear(n_features2, n_features1),
        fc_layer1,
    )
    models = [model12, model21]
    ground_truth = [[model1, fc_layer1], [model2, fc_layer2]]

    # acquire data
    train_data_loader = get_dataloader(
        dataset_dir="",  # not required for cifar100
        dataset_name="cifar100",
        image_size=384,
        batch_size=16,
        num_workers=2,
        is_distributed=False,
        seed=123,
        return_original_image=False,
    )

    # train both models
    for i_model, model in enumerate(models):
        criterion = torch.nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_data_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                (
                    inputs,
                    _labels,
                ) = data  # we ignore ground truth labels because what interests us here is what the original model would have done

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(
                    outputs, ground_truth[i_model][1](ground_truth[i_model][0](inputs))
                )
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 100 == 99:  # print every 2000 mini-batches
                    print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                    running_loss = 0.0
        output_path.mkdir(exist_ok=True, parents=True)
        torch.save(model, output_path / i_model)

    print("Finished Training")


if __name__ == "__main__":
    load_and_pretrain_networks()
    train_linear()
