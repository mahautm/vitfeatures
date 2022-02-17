# load one model
# load another
# chop off the classif layers; switch them. add a linear mapping in between
# make sure grad is only activated for the new linear layer
# load a dataset (does it have to be images or can it be just any random image ?)
# for each image train only one linear layer to map one to the other


from pyexpat import model
from data import get_dataloader
from archs import initialize_vision_module
from torch import nn
from torch.nn import Sequential
import torch.optim as optim

n_epochs = 10

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
train_data_loader = get_dataloader(
    dataset_name="cifar100",
    image_size="384",
    batch_size=16,
    num_workers=2,
    is_distributed=False,
    seed=123,
    return_original_image=False,
)


for model in models:
    criterion = nn.CrossEntropyLoss()  # crosent ?
    optimizer = optim.SGD(
        model.parameters(), lr=0.001, momentum=0.9
    )  # SGD ? adam is better no ?
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

print("Finished Training")
