import torch


def test_models(model1, model2, model21, model12, train_data_loader):
    """
    LossA,  = Acc(CA(FA())) − Acc(CA(MB→AFB()))
    """
    with torch.no_grad():
        # test all models on all data
        accs = []
        for model in [model1, model2, model21, model12]:
            accuracy = [model(input) == label for input, label in train_data_loader]
            accs.append(sum(accuracy) / len(accuracy))
        print(
            f"training accuracy : m1 = {accs[0]}, m2 = {accs[1]}, m21 = {accs[2]}, m12 = {accs[3]},"
        )
        print(f"loss12 = {accs[0] - accs[2]}, loss21 = {accs[1] - accs[3]}")
    pass
