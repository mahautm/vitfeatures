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

model1, fc_layer1, n_features1 = initialize_vision_module("vit")
model2, fc_layer2, n_features2 = initialize_vision_module("resnet152")

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
train_data_loader = get_dataloader(
    dataset_dir="",  # not required for cifar100
    dataset_name="cifar100",
    image_size="384",
    batch_size=16,
    num_workers=2,
    is_distributed=False,
    seed=123,
    return_original_image=False,
)

for i, model in enumerate(models):
    criterion = torch.cdist
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            (
                inputs,
                _,
            ) = data  # we ignore ground truth labels because what interests us here is what the original model would have done

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, ground_truth[i][1](ground_truth[i][0](inputs)))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0
    output_path.mkdir(exist_ok=True, parents=True)
    torch.save(model, output_path / i)

print("Finished Training")
