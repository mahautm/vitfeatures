# Usage:
# python ./compute_rsa_on_resnet_outputs.py \
# /private/home/rdessi/interactions_for_marco/latest_interaction_for_neurips_all_fields_valid
import os, sys
import torch
import numpy as np
from sklearn import metrics

# from scipy.stats import spearmanr
from scipy.stats import pearsonr
from archs import initialize_vision_module, behead_freeze
from data import get_dataloader


device = "cuda"
# interaction_path = sys.argv[1]
# print(f"loading {interaction_path} and reading in data")
# interaction = torch.load(interaction_path)
# sender_res = interaction.aux["resnet_output_sender"].numpy()
# receiver_res = interaction.aux["resnet_output_recv"].numpy()

# print("computing pairwise cosines")
# sender_cos = metrics.pairwise.cosine_similarity(sender_res)
# receiver_cos = metrics.pairwise.cosine_similarity(receiver_res)

# print("extracting uppper triangular matrices")
# sender_upper_tri = sender_cos[np.triu_indices(sender_cos.shape[0], k=1)]
# receiver_upper_tri = receiver_cos[np.triu_indices(receiver_cos.shape[0], k=1)]
# print("computing Pearson and Spearman correlations")
# print(f"Pearson correlation: {pearsonr(sender_upper_tri,receiver_upper_tri)[0]}")
# # print(f"Spearman correlation: {spearmanr(sender_upper_tri,receiver_upper_tri).correlation}")

# v2.rsa_check('./models/vgg11','./models/vit')
def rsa_check(
    model_path="./models",
    model_name_1="vgg11",
    model_name_2="vit",
    verbose=False,
    seed=123,
):
    # load trained models
    train_data_loader = get_dataloader(
        dataset_dir="./data",
        dataset_name="cifar100",
        image_size=384,
        batch_size=64,
        num_workers=2,
        seed=seed,
        return_original_image=False,
    )
    # model1, model2 = torch.load(os.path.join(model_path, model_name_1)), torch.load(
    #     os.path.join(model_path, model_name_2)
    # )
    model1, model2 = initialize_vision_module(
        model_name_1, pretrained=True
    ), initialize_vision_module(model_name_2, pretrained=True)
    model1, _, _ = behead_freeze(model1, model_name_1)
    model2, _, _ = behead_freeze(model2, model_name_2)
    f1_cos, f2_cos, pearsonr = compute_model_rsa(
        train_data_loader, model1, model2, verbose=verbose
    )
    return (f1_cos, f2_cos, pearsonr)
    # store data / make some kind of graph


def compute_model_rsa(train_data_loader, model1, model2, n_images=10000, verbose=False):
    features = [[], []]

    for i in range(n_images // train_data_loader.batch_size):
        images, _ = next(iter(train_data_loader))
        for model_number, model in enumerate([model1.to(device), model2.to(device)]):
            if verbose:
                print(
                    f"model: {model_number +1} batch: {i + 1}/{n_images // train_data_loader.batch_size}"
                )
            feature = model(images.to(device))
            if i == 0:
                features[model_number] = feature.to("cpu").detach().numpy()
            else:
                np.concatenate(
                    (features[model_number], feature.to("cpu").detach().numpy())
                )
    # print("computing pairwise cosines")
    f1_cos = metrics.pairwise.cosine_similarity(features[0])
    f2_cos = metrics.pairwise.cosine_similarity(features[1])
    # print("extracting uppper triangular matrices")
    f1_upper_tri = f1_cos[np.triu_indices(f1_cos.shape[0], k=1)]
    f2_upper_tri = f2_cos[np.triu_indices(f2_cos.shape[0], k=1)]
    # print("computing Pearson and Spearman correlations")
    return f1_cos, f2_cos, pearsonr(f1_upper_tri, f2_upper_tri)[0]


if __name__ == "__main__":
    params = sys.argv[1:]  # TODO : argparse
    print(params)
    _, _, rsa_score = rsa_check(
        model_name_1=params[0],
        model_name_2=params[1],
        seed=params[2] if len() > 2 else 123,
    )
    print(rsa_score)
