# Usage:
# python ./compute_rsa_on_resnet_outputs.py \
# /private/home/rdessi/interactions_for_marco/latest_interaction_for_neurips_all_fields_valid
import sys
import torch
import numpy as np
from sklearn import metrics

# from scipy.stats import spearmanr
from scipy.stats import pearsonr

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
def compute_model_rsa(train_data_loader, model1, model2, n_images=10000, verbose=True):
    # print("reading data")
    features = [[], []]
    for model_number, model in enumerate([model1, model2]):
        model = model.to(device)
        for i in range(n_images // train_data_loader.batch_size):
            if verbose:
                print(
                    f"model: {model_number} batch: {i}/{n_images // train_data_loader.batch_size}"
                )
            images, _ = next(iter(train_data_loader))
            features[model_number].append(model(images.to(device)).to("cpu"))
            # maybe here add garbage collection
            torch.cuda.empty_cache()
    # print("computing pairwise cosines")
    f1_cos = metrics.pairwise.cosine_similarity(features[0])
    f2_cos = metrics.pairwise.cosine_similarity(features[1])
    # print("extracting uppper triangular matrices")
    f1_upper_tri = f1_cos[np.triu_indices(f1_cos.shape[0], k=1)]
    f2_upper_tri = f2_cos[np.triu_indices(f2_cos.shape[0], k=1)]
    # print("computing Pearson and Spearman correlations")
    return f1_cos, f2_cos, pearsonr(f1_upper_tri, f2_upper_tri)[0]
