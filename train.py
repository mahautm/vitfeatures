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

train_loader = get_dataloader(
    dataset_name="cifar100",
    image_size="384",
    batch_size=16,
    num_workers=2,
    is_distributed=False,
    seed=123,
    return_original_image=False,
)
