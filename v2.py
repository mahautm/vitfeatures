# load a dataset (does it have to be images or can it be just any random image ?)
# for each image train only one linear layer to map one to the other

# Sooo thoughts here. We train on the other network's answers on not on the ground truth (a bit like distillation ?)
# Does this have consequences ?
# Also : they use the dataset that was used to train the original networks. Does it make sense to use another one ? (here cifar because easiest)

from data import get_dataloader
from archs import initialize_vision_module, behead_freeze
from train import train_model
from test import test_models
from rsa import compute_model_rsa
import torch
from torch import nn
from torch.nn import Sequential
import torch.optim as optim
import os

# PARAMS
output_path = "/shared/mateo/vitfeatures"
device = "cuda"


def get_models(
    train_data_loader=None,
    model_name_1=None,
    model_name_2=None,
):
    model1 = initialize_vision_module(model_name_1, pretrained=True)
    model2 = initialize_vision_module(model_name_2, pretrained=True)
    # optimiser1 = optim.Adam(model1.parameters(), lr=0.0001)
    # optimiser2 = optim.Adam(model2.parameters(), lr=0.0001)
    # model1 = train_model(
    #     model1,
    #     train_data_loader,
    #     optimiser1,
    #     criterion=torch.nn.CrossEntropyLoss(),
    #     verbose=True,
    # )
    # model2 = train_model(
    #     model2,
    #     train_data_loader,
    #     optimiser2,
    #     criterion=torch.nn.CrossEntropyLoss(),
    #     verbose=True,
    # )
    # print("Finished Initial Training")

    # # Save models
    os.makedirs(output_path, exist_ok=True)
    torch.save(model1, output_path + "/" + model_name_1)
    torch.save(model2, output_path + "/" + model_name_2)

    return model1, model2


def features(model_name_1="vit", model_name_2="vgg11"):
    """
    TODO : n_epochs is implicitly set at 10 during training, think on that
    TODO : validation must be added during training, as well as better logging
    """
    # get data
    test_data_loader, train_data_loader = get_dataloader(
        dataset_dir="",  # not required for cifar100
        dataset_name="cifar100",
        image_size=384,
        batch_size=2,
        num_workers=2,
        seed=123,
        return_original_image=False,
    )

    # get and train models
    model1, model2 = get_models(model_name_1=model_name_1, model_name_2=model_name_2)

    # get features from one model to be linked by a linear layer to the other's classification layer
    feature_extractor_1, classifier_1, n_features1 = behead_freeze(model1, model_name_1)
    feature_extractor_2, classifier_2, n_features2 = behead_freeze(model2, model_name_2)

    mapper12 = Sequential(
        feature_extractor_1,
        nn.Linear(n_features1, n_features2),
    )

    mapper21 = Sequential(
        feature_extractor_2,
        nn.Linear(n_features2, n_features1),
    )

    # train the linear layer to map features to classifiers
    optimiser1 = optim.Adam(mapper12.parameters(), lr=0.0001)
    optimiser2 = optim.Adam(mapper21.parameters(), lr=0.0001)
    mapper12 = train_model(
        mapper12,
        train_data_loader,
        label_model=feature_extractor_2,
        optimiser=optimiser1,
    )
    mapper21 = train_model(
        mapper21,
        train_data_loader,
        label_model=feature_extractor_1,
        optimiser=optimiser2,
    )
    print("Finished Linear Mapping")

    # analyse hybrid model performance
    test_models(
        model1,
        model2,
        Sequential(mapper12, classifier_2),
        Sequential(mapper21, classifier_1),
        train_data_loader,
        test_data_loader,
    )
    # save hybrids
    torch.save(mapper12, output_path / model_name_1 + "lin")
    torch.save(mapper21, output_path / model_name_2 + "lin")
    pass


def rsa_check(model1_path, model2_path):
    # load trained models
    _, train_data_loader = get_dataloader(
        dataset_dir="",  # not required for cifar100
        dataset_name="cifar100",
        image_size=384,
        batch_size=1,
        num_workers=2,
        seed=123,
        return_original_image=False,
    )
    model1, model2 = torch.load(model1_path), torch.load(model2_path)
    f1_cos, f2_cos, pearsonr = compute_model_rsa(train_data_loader, model1, model2)
    print(f1_cos, f2_cos, pearsonr)
    # store data / make some kind of graph
    pass


def reproduce_paper(model_name_1="vit", model_name_2="vgg11"):
    # get data
    test_data_loader, train_data_loader = get_dataloader(
        dataset_dir="",  # not required for cifar100
        dataset_name="cifar100",
        image_size=384,
        batch_size=16,
        num_workers=2,
        seed=123,
        return_original_image=False,
    )
    """
    TODO : n_epochs is implicitly set at 10 during training, think on that
    TODO : validation must be added during training, as well as better logging
    """
    # get and train models
    model1 = initialize_vision_module(model_name_1)
    model2 = initialize_vision_module(model_name_2)
    model1 = train_model(model1, train_data_loader)
    model2 = train_model(model2, train_data_loader)
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
    )

    model21 = Sequential(
        feature_extractor_2,
        nn.Linear(n_features2, n_features1),
        classifier_1,
    )

    # train the linear layer to map features to classifiers
    model12 = train_model(model12, train_data_loader, label_model=model2)
    model21 = train_model(model21, train_data_loader, label_model=model1)
    print("Finished Linear Mapping")

    # analyse hybrid model performance
    test_models(model1, model2, model21, model12, train_data_loader, test_data_loader)
    # save hybrids
    torch.save(model12, output_path / model_name_1)
    torch.save(model21, output_path / model_name_2)


if __name__ == "__main__":
    features()
