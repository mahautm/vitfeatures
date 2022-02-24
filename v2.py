# load a dataset (does it have to be images or can it be just any random image ?)
# for each image train only one linear layer to map one to the other

# Sooo thoughts here. We train on the other network's answers on not on the ground truth (a bit like distillation ?)
# Does this have consequences ?
# Also : they use the dataset that was used to train the original networks. Does it make sense to use another one ? (here cifar because easiest)

from data import get_dataloader
from archs import initialize_vision_module, behead_freeze
from train import train_model
from test import test_models
import torch
from torch import nn
from torch.nn import Sequential

# PARAMS
output_path = "/mnt/efs/fs1/logs/vitfeatures"
device = "cuda"


def full_experiment(model_name_1="vit", model_name_2="vgg11"):
    # get data
    test_data_loader, train_data_loader = get_dataloader(
        dataset_dir="",  # not required for cifar100
        dataset_name="cifar100",
        image_size=384,
        batch_size=16,
        num_workers=2,
        is_distributed=False,
        seed=123,
        return_original_image=False,
    )
    """
    TODO : n_epochs is implicitly set at 10 during training, think on that
    TODO : validation must be added during training, as well as better logging
    """
    # get and train models
    model1 = initialize_vision_module(model_name_1).to(device)
    model2 = initialize_vision_module(model_name_2).to(device)
    model1 = train_model(model1)
    model2 = train_model(model2)
    print("Finished Initial Training")
    
    # Save models
    output_path.mkdir(exist_ok=True, parents=True)
    torch.save(model1, output_path / model_name_1)
    torch.save(model2, output_path / model_name_2)

    # get features from one model to be linked by a linear layer to the other's classification layer
    feature_extractor_1, classifier_1, n_features1 = behead_freeze(model1, model_name_1)
    feature_extractor_2, classifier_2, n_features2 = behead_freeze(model2, model_name_2)

    model12 = Sequential(
        feature_extractor_1,
        nn.Linear(n_features1, n_features2),
        classifier_2,
    ).to(device)

    model21 = Sequential(
        feature_extractor_2,
        nn.Linear(n_features2, n_features1),
        classifier_1,
    ).to(device)

    # train the linear layer to map features to classifiers
    model12 = train_model(model12, train_data_loader, label_model=model2)
    model21 = train_model(model21, train_data_loader, label_model=model1)
    print("Finished Linear Mapping")

    # analyse hybrid model performance
    test_models(model1, model2, model21, model12)
    # save hybrids
    torch.save(model1, output_path / model_name_1)
    torch.save(model2, output_path / model_name_2)



if __name__ == "__main__":
    pass
