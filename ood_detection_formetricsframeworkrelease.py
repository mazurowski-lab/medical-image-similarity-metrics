# see if different feature representations can be used for OOD detection
import torch
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import scipy.stats as sps
import os
from sklearn.metrics import roc_auc_score

from src.radiomics.radiomics_utils import convert_radiomic_dfs_to_vectors, compute_and_save_imagefolder_radiomics_parallel
from src.utils import *
from src.dataset import SimpleImageDataset

from argparse import ArgumentParser

device = 'cuda' if torch.cuda.is_available() else 'cpu'
viz_folder = 'visualizations'
img_size = 256

import random
# set random seed
fix_seed = True
if fix_seed:
    seed = 1338
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def auc_deviation(labels, scores):
    return 2. * (roc_auc_score(labels, scores) - 0.5)

def get_imagenet_features(
    in_img_folder,
    out_img_folder_id,
    out_img_folder_ood,
    weights,
    bs = 16
):
    # load pretrained inception v3 model for feature inversion
    if weights == 'ImageNet':
        model = models.inception_v3(pretrained=True)
    elif weights == 'RadImageNet':
        print("\n\n USING RADIMAGENET WEIGHTS to COMPUTE! \n\n")
        # convert their checkpoint to torchvision model
        path = 'src/gan-metrics-pytorch/models/RadImageNet_InceptionV3.pt'
        base_model = models.inception_v3(pretrained=False, aux_logits=False)
        encoder_layers = list(base_model.children())
        # print children names
        backbone = nn.Sequential(*encoder_layers[:-1])

        new_names = [n for n,_ in base_model.named_children()][:-1]
        original_names = [n for n,_ in backbone.named_children()]

        original_state_dict = torch.load(path)
        new_state_dict = {}
        for k, v in original_state_dict.items():
            new_k = k.replace("backbone.", "")
            old_name = new_k.split(".")[0]
            new_name = new_names[original_names.index(old_name)]
            new_k = new_k.replace(old_name, new_name)
            new_state_dict[new_k] = v

        base_model.load_state_dict(new_state_dict, strict=False)
        model = base_model
    else:
        raise ValueError('Invalid weights parameter. Must be "ImageNet" or "RadImagenet".')

    model.eval().to(device)

    # create a list to store the activations
    activations = []
    def get_activation(lyr_name):
        def hook(model, input, output):
            activations.append(output.detach())
        return hook

    # register hook
    # want to use final pooling layer as feature (pool3)
    lyr_name = 'avgpool'
    lyr = model.avgpool
    hook = lyr.register_forward_hook(get_activation(lyr_name))

    # create simple dataloaders for input image dataset and output image dataset
    # first create torch datasets
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    image_mode = 'RGB'
    in_set = SimpleImageDataset(in_img_folder, image_mode, transform)
    out_set_id = SimpleImageDataset(out_img_folder_id, image_mode, transform)
    out_set_ood = SimpleImageDataset(out_img_folder_ood, image_mode, transform)

    in_loader = DataLoader(in_set, 
                    batch_size=bs)
    out_loader_id = DataLoader(out_set_id, 
                    batch_size=bs)
    out_loader_ood = DataLoader(out_set_ood,
                    batch_size=bs)

    # get features
    in_activations = []
    out_activations_id = []
    out_activations_ood = []
    for batch in tqdm(in_loader):
        inputs = batch
        # Copy inputs to device
        inputs = inputs.to(device)

        # get final conv-layer activations for batch
        outputs = model(inputs)

        in_activations.append(activations[0].cpu())
        activations = []

    for batch in tqdm(out_loader_id):
        inputs = batch
        # Copy inputs to device
        inputs = inputs.to(device)

        # get final conv-layer activations for batch
        outputs = model(inputs)

        out_activations_id.append(activations[0].cpu())
        activations = []
    
    for batch in tqdm(out_loader_ood):
        inputs = batch
        # Copy inputs to device
        inputs = inputs.to(device)

        # get final conv-layer activations for batch
        outputs = model(inputs)

        out_activations_ood.append(activations[0].cpu())
        activations = []

    in_activations = torch.cat(in_activations, dim=0).squeeze()
    out_activations_id = torch.cat(out_activations_id, dim=0).squeeze()
    out_activations_ood = torch.cat(out_activations_ood, dim=0).squeeze()

    return in_activations, out_activations_id, out_activations_ood

def main(
        in_img_folder,
        out_img_folder_id,
        out_img_folder_ood,
        features_name = 'RadImageNet',
        val_frac = 0.1,
        measure_individual_nRaDs = True,
        use_val_set = False
):

    if "ImageNet" in features_name:
        in_activations, out_activations_id, out_activations_ood = get_imagenet_features(
            in_img_folder,
            out_img_folder_id,
            out_img_folder_ood,
            features_name
        )
    elif features_name == "Radiomics":
        radiomics_path1 = os.path.join(in_img_folder, 'radiomics.csv')
        radiomics_path2_id = os.path.join(out_img_folder_id, 'radiomics.csv')
        radiomics_path2_ood = os.path.join(out_img_folder_ood, 'radiomics.csv')

        # if needed, compute radiomics for the images
        if not os.path.exists(radiomics_path1):
            print("No radiomics found computed for image folder 1 at {}, computing now.".format(radiomics_path1))
            compute_and_save_imagefolder_radiomics_parallel(in_img_folder)
            print("Computed radiomics for image folder 1.")
        else:
            print("Radiomics already computed for image folder 1 at {}.".format(radiomics_path1))

        if not os.path.exists(radiomics_path2_id):
            print("No radiomics found computed for image folder 2 at {}, computing now.".format(radiomics_path2_id))
            compute_and_save_imagefolder_radiomics_parallel(out_img_folder_id)
            print("Computed radiomics for image folder 2.")
        else:
            print("Radiomics already computed for image folder 2 at {}.".format(radiomics_path2_id))

        if not os.path.exists(radiomics_path2_ood):
            print("No radiomics found computed for image folder 3 at {}, computing now.".format(radiomics_path2_ood))
            compute_and_save_imagefolder_radiomics_parallel(out_img_folder_ood)
            print("Computed radiomics for image folder 3.")
        else:
            print("Radiomics already computed for image folder 3 at {}.".format(radiomics_path2_ood))

        # load radiomics
        radiomics_df1 = pd.read_csv(radiomics_path1)
        radiomics_df2_id = pd.read_csv(radiomics_path2_id)
        radiomics_df2_ood = pd.read_csv(radiomics_path2_ood)

        # print shape of radiomics dataframes
        #print(radiomics_df1.shape, radiomics_df2_id.shape, radiomics_df2_ood.shape)

        in_activations, out_activations_id, id_ref_filenames, id_filenames = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                             radiomics_df2_id,
                                                             match_sample_count=True, # needed for distance measures
                                                             return_image_fnames=True
                                                             ) 

        _, out_activations_ood, _, ood_filenames = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                             radiomics_df2_ood,
                                                             match_sample_count=True, # needed for distance measures
                                                             return_image_fnames=True
                                                             ) 


    # randomly split in_activations into train and val, via val_frac
    val_idx = np.random.choice(in_activations.shape[0], int(val_frac*in_activations.shape[0]), replace=False)
    train_idx = np.array([i for i in range(in_activations.shape[0]) if i not in val_idx])

    in_activations_val = in_activations[val_idx]
    in_activations = in_activations[train_idx]

    if not use_val_set:
        in_activations_val = in_activations
        print("Using training set as validation set.")

    in_activations = torch.tensor(in_activations)
    out_activations_id = torch.tensor(out_activations_id)
    out_activations_ood = torch.tensor(out_activations_ood)

    id_mean = in_activations.mean(dim=0)

    #assert False

    # AUC analysis for OOD detection
    labels = np.concatenate([np.zeros(len(out_activations_id)), np.ones(len(out_activations_ood))])

    # scores are L2 between mean of in_activations and each out_activations
    scores = torch.stack([torch.norm(id_mean - out, dim=0) for out in torch.cat([out_activations_id, out_activations_ood])])
    scores = scores.detach().numpy()

    try:
        all_img_filenames = np.concatenate([id_filenames, ood_filenames])
    except UnboundLocalError as e:
        print(e)
        all_img_filenames = [None for s in scores]
        pass

    auc = roc_auc_score(labels, scores)
    print(f"Threshold-independent AUC: {auc}")

    # further analysis
    ID_scores = scores[:len(out_activations_id)]
    OOD_scores = scores[len(out_activations_id):]

    ID_scores_val = torch.stack([torch.norm(id_mean - out, dim=0) for out in in_activations_val])
    ID_scores_val = ID_scores_val.detach().numpy()
    # find OOD detection threshold via dist of in-distribution validation set to in dist training set
    # attmpt this using statistical testing and Gaussian assumption
    mu, sigma = np.mean(ID_scores_val), np.std(ID_scores_val)

    # DOF for t-test
    dof = len(ID_scores_val) - 1

    for ID_dist_assumption in ["counting"]:#, "t"]:
        if ID_dist_assumption == 'gaussian':
            threshOOD = sigma*sps.norm.ppf(0.95) + mu

            # Calculate z-score
            z = (OOD_scores - mu) / sigma

            # Calculate the p-values
            p_value = 1 - sps.norm.cdf(z)  # One-tailed test

        elif ID_dist_assumption == 't':
            threshOOD = sigma*sps.t.ppf(0.95, dof) + mu

            # Calculate t-score
            t = (OOD_scores - mu) / (sigma / np.sqrt(len(ID_scores_val)))

            # Calculate the p-values
            p_value = 1 - sps.t.cdf(t, dof)

        elif ID_dist_assumption == "counting":
            # get threshold by counting
            threshOOD = np.percentile(ID_scores_val, 95)

            # Calculate p-value by counting
            p_value = np.array([np.sum(ID_scores_val > score) / len(ID_scores_val) for score in OOD_scores])

        # compute OOD detection accuracy when using the 95th percentile of the in-distribution scores as threshold
        threshold = threshOOD
        print("Metrics using automatic threshold of {}:".format(threshold))
        pred = scores > threshold
        acc = np.mean(pred == labels)

        tpr = np.sum((pred == 1) & (labels == 1)) / np.sum(labels == 1)
        fpr = np.sum((pred == 1) & (labels == 0)) / np.sum(labels == 0)
        tnr = 1 - fpr
        fnr = 1 - tpr

        print('\nAccuracy = {}\nAUC = {}\nSensitivity/TPR = {}\nSpecificity/TNR = {}'.format(round(acc, 3), round(auc, 3), round(tpr, 3), round(tnr, 3)))


if __name__ == '__main__':
    IMAGE_FOLDER1="data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/train/GE"
    IMAGE_FOLDER2_id="data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/GE"
    IMAGE_FOLDER2_ood="data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/Siemens"

    #IMAGE_FOLDER1='data/brats/trainB'
    #IMAGE_FOLDER2_id='data/brats/testB'
    #IMAGE_FOLDER2_ood='data/brats/testA'

    #IMAGE_FOLDER2_id='data/brats/testB_seppatients'
    #IMAGE_FOLDER2_ood='data/brats/testA_seppatients'

    #IMAGE_FOLDER1='data/lumbar_mritoct/CT/images/train'
    #IMAGE_FOLDER2_id='data/lumbar_mritoct/CT/images/test'
    #IMAGE_FOLDER2_ood='data/lumbar_mritoct/MRI/images/test'

    #IMAGE_FOLDER1='data/chaos/split_by_domain/images/trainB'
    #IMAGE_FOLDER2_id='data/chaos/split_by_domain/images/testB'
    #IMAGE_FOLDER2_ood='data/chaos/split_by_domain/images/testA'

    main(
        IMAGE_FOLDER1, 
        IMAGE_FOLDER2_id,
        IMAGE_FOLDER2_ood,
        #features_name = 'Radiomics'
        # features_name = 'RadImageNet'
        #features_name = 'ImageNet'
        )
