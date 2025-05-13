"""
analyze and explore domain shift: train, evaluate and interpret models
"""
import os
import sys

from src.dataset import *
from src.utils import *
from src.vizutils import *
from src.adversarial_utils import *
from src.radiomics.radiomics_utils import compute_slice_radiomics, convert_radiomic_dfs_to_vectors

import os
import random
from tqdm import tqdm
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
from PIL import Image
from pytorch_fid import fid_score
from sklearn.metrics import roc_auc_score

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

#import lucent
#from lucent.optvis import render, param, transform, objectives
#from lucent.misc.io import show

# torch
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, swin_t
from unet import UNet2D
from torchvision.utils import save_image

from monai.losses import DiceLoss

# GPUs
#device_ids = [0] # indices of devices for models, data and otherwise
#os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
print('running on {}'.format(device))

# set random seed
fix_seed = True
if fix_seed:
    seed = 1338
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

dataset_name = 'dbc_priorwork1k'
#dataset_name = 'dbc_by_cancer'
# old 
# dataset_name = 'dbc_by_scanner_reconstructions'
# dataset_name = 'dbc_by_scanner'

#dataset_name = 'gmsc_sites24'


print("using dataset: {}".format(dataset_name))

# data and model choice
labeling = 'default' # default is cancer classification (or the default segmentation task is for segmentation)

#labeling = 'feature: Lymphadenopathy or Suspicious Nodes'
#labeling = 'feature: Manufacturer'

# only work with images that have certain  value for a certain IAP
test_this_IAP_only = None
#test_this_IAP_only = {'feature: Manufacturer' : 0}

if test_this_IAP_only is not None:
    print("only using images with IAP {}".format(test_this_IAP_only))

# only look at ROI (ie no background pixels) 
roi_only = False
balance_classes = False

if roi_only:
    print("\n\nonly using ROI in images!!!\n\n")

#if labeling == 'feature: Manufacturer':
#    # for domain-differing feature studies
#    if dataset_name.startswith('dbc'):
#        print("only using ROI in images")
#        roi_only = True

# v trained on ROI only
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/resnet18_11996_2410_0.9457610099988766_3_feature: Manufacturer_best.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/resnet18_3356_696_0.6471619285439878_3_default_best_trainedon_domain0.h5"

# train on normal images
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/resnet18_11996_2410_1.0_0_feature: Manufacturer_best.h5"

# fat intensity augmentation exps
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11996_2410_0.6403121092728243_0_default_best_aug_trainedon_domain0.h5"

# TR stratification exps

#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11996_2410_0.7239896839610581_8_default_best_trainedon_domain0_fgt_lowTR.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11996_2410_0.6868441196607088_15_default_best_trainedon_domain0_fgt_highTR.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11996_2410_0.3805101721669637_4_default_best_trainedon_domain1_fgt_lowTR.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11996_2410_0.3185606009363468_0_default_best_trainedon_domain1_fgt_highTR.h5"

# v OLD v
#checkpoint_path = "saved_models/feature_pred/dbc_by_cancer/resnet18_9872_1974_0.7637367219795892_6_default_best_trainedon_domain0.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_by_cancer/resnet18_9872_1974_0.7426094969716801_5_default_best_trainedon_domain1.h5"

#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11366_2275_0.8897227407795202_11_default_best_trainedon_domain0_breast.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11366_2275_0.9373278366008275_20_default_best_trainedon_domain1_breast.h5"

#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11366_2275_0.6744995937527546_29_default_best_trainedon_domain0_fgt.h5"
#checkpoint_path = "saved_models/feature_pred/dbc_priorwork1k/UNet2D_11366_2275_0.6463053992516558_15_default_best_trainedon_domain1_fgt.h5"

# GMSC v
#checkpoint_path = "saved_models/feature_pred/gmsc_sites24/resnet18_151_46_1.0_4_feature: Manufacturer_best.h5"

#checkpoint_path = "saved_models/feature_pred/gmsc_sites24/UNet2D_151_46_0.9171643853187561_45_feature: Manufacturer_best_trainedon_domain0.h5"
#checkpoint_path = "saved_models/feature_pred/gmsc_sites24/UNet2D_151_46_0.9374296069145203_88_feature: Manufacturer_best_trainedon_domain1.h5"


# OLD
# if not using pre-defined splits
# dbc_by_* datasets
# train_size = 10000
# val_size = 2000
# test_size = 2000

# for training on reconstructions
# train_size = 1000
# val_size = 1000
# test_size = 1

#task = 'classification'
# task = 'regression'
task = 'segmentation'

print("task: {}".format(task))

# segmentation options
target_mask_batch_index = -1 # for breast MRI: -2 = breast mask, -1 = FGT + BV mask

mask_class = None # only predict mask for this class
if task == 'segmentation':
    img_size = 256 # see https://github.com/fepegar/unet/issues/28
    model = UNet2D
    if dataset_name.startswith("dbc"):
        if target_mask_batch_index == -2: # breast mask
            mask_class = 1 # only predict mask for this class
        elif target_mask_batch_index == -1: #FGT + BV mask
            mask_class = 2 # FGT
            if mask_class == 1:
                print("NOTICE! segmenting BV, NOT FGT.")
        else:
            raise NotImplementedError

        num_unet_encoding_blocks = 5
    else:
        mask_class = 1
        num_unet_encoding_blocks = 3

else:
    img_size = 224
    model = resnet18
    # model = TinyConv

class_counts = {
   'default' : 2, # cancer classification
   'feature: Manufacturer' : 3 ,
   'feature: Manufacturer Model Name' : 8,
   'feature: Scan Options' : 9,
   'feature: Patient Position During MRI' : 2,
   'feature: Field Strength (Tesla)' : 4,
   'feature: Contrast Agent' : 6,
#    'feature: Contrast Bolus Volume (mL)': 19,
   'feature: Acquisition Matrix': 10,
   'feature: Slice Thickness ': 21,
   #'feature: Reconstruction Diameter ': 20,
   'feature: Flip Angle \n': 4,
   'feature: FOV Computed (Field of View) in cm ': 27,

   'feature: Breast Density': 6,
   'feature: Lymphadenopathy or Suspicious Nodes': 2
}
if task == 'classification':
    num_classes = class_counts[labeling]

tgt_bidx = target_mask_batch_index if task == 'segmentation' else 1

# training options
train = True
epochs = 100
batch_size_factors = {
          'resnet18' : 64,
          'swin_t' : 64,
          'UNet2D' : 8,
          'TinyConv' : 16
}

bs_eval = 16

lrs = {
          'resnet18' : 0.001, # for SC # 0.00001, # for DBC
          'swin_t' : 0.0001,
          'UNet2D' : 0.01,
          'TinyConv' : 0.01
}
lr = lrs[model.__name__]

checkpoint_path_prev = None
train_with_augmentations = True
save_checkpoints = True
early_stopping = True
early_stopping_epochs = 5

evaluate_domain_shift = True # doesnt affect feature viz stuff
if evaluate_domain_shift:
    source_domain_label = 0
    target_domain_label = 1 - source_domain_label
    print("evaluating domain shift scenario. source domain: {}, target domain: {}".format(source_domain_label, target_domain_label))

# more domain shift options
use_TR_only = None
#use_TR_only = "low" #None
#use_TR_only = "high" #None

if use_TR_only is not None:
    assert dataset_name.startswith("dbc")
    print("only using {} TR images".format(use_TR_only))

# evaluation options
reload_net = not train
# reload_net = False
save_testset_predictions = False

full_test = False
adv_attack = False
do_gradcam = False
do_attributions = False
evaluate_feature_space = False
evaluate_fourier = False

save_individual_losses = False
if save_individual_losses:
    print("saving individual losses on test set images")

evaluate_harmonization = False
harmonized_dataset_name = ['dbc_priorwork1k_test_harmonized_0to1_PWlinear_breast_0.0', 'data/dbc/prior_work_1k/harmonized/by_piecewise_linear_breast/0to1/level0.0', ['data/dbc/prior_work_1k/segmentations2D/breast/test', 'data/dbc/prior_work_1k/segmentations2D/dv/test']]

#harmonized_dataset_name = 'dbc_priorwork1k_test_GE'
#harmonized_dataset_name = 'dbc_priorwork1k_test_Siemens_corrupted'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_cyclegan'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_cyclegan_corrupted'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_maskGAN'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_MUNIT'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_CUT'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_UNSB'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_GCGAN'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_uncondDDIM'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_contourDDIM'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_SPADE'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_segguidedDDIM'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_SiemenstoGE_PWlinear'

#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_GEtoSiemens_cyclegan'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_GEtoSiemens_uncondDDIM'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_GEtoSiemens_contourDDIM'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_GEtoSiemens_SPADE'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_GEtoSiemens_segguidedDDIM'
#harmonized_dataset_name = 'dbc_priorwork1k_test_harmonized_GEtoSiemens_PWlinear'

# additional harmonization metrics besides chosen downstream task performance
eval_harmonization_radiomics = False
# options
realdata_radiomics_fname = "data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/GE/radiomics.csv"
#realdata_radiomics_fname = "data/dbc/prior_work_1k/mri_data_labeled2D/test/radiomics_domain0_breast.csv"
#realdata_radiomics_fname = "data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/GE/radiomics_foreground.csv"

use_segmentation_radiomics = False
use_foreground_radiomics = False
assert not (use_segmentation_radiomics and use_foreground_radiomics)

use_textural_radiomics_only = False
use_wavelet_radiomics_only = False
use_simple_radiomics_only = False

# separate option to run simple harmonization method,
# controlled by source_domain_label and target_domain_label
also_harmonize_linearly_w_radiomics = False

flip_these_domain_labels = None
#flip_these_domain_labels = '/shared/for_yuwen/cyclegan_baseline/resnet_9blocks-basic-0.001-1/test_samples/BtoA'

if evaluate_harmonization:
    assert full_test
    assert not evaluate_domain_shift, "evaluate domain shift only used during model training for evaluation"

# experimental options
corrupt_images = False # set bottom threshold of intensities to zero
only_corrupt_test_images = False

#image_corruption_mode = 'smoothing'
noise_level = 3

image_corruption_mode = 'single_pixel'
#random_pix_ind = 0 #np.random.randint(0, img_size)
random_pix_ind = 168 #np.random.randint(0, img_size)

#image_corruption_mode = 'percentile'
background_removal_frac = 0.01


if corrupt_images:
    print("corrupting images.")

# load dataset and loader
train_batchsize = batch_size_factors[model.__name__] # * len(device_ids)
if do_attributions:
    train_batchsize = 200
    bs_eval = train_batchsize


train_subset = None
if train_subset is not None:
    print("\n\n\n!!using subset of training set!!\n\n\n")

# old
# if dataset_name.startswith("dbc_by"):
#     loaded_dataset_name = dataset_name
#     if evaluate_harmonization:
#         loaded_dataset_name = harmonized_dataset_name
#         print("using harmonized dataset {}".format(loaded_dataset_name))

#     trainset, valset, testset = get_datasets(loaded_dataset_name, 
#                                     train_size=train_size, 
#                                     val_size=val_size,
#                                     test_size=test_size,
#                                     labeling=labeling,
#                                     task=task,
#                                     return_filenames=True,
#                                     return_filepaths=True,
#                                     different_cases_for_train_val_test=True,
#                                     roi_only=roi_only,
#                                     balance_classes=balance_classes,
#                                     test_this_IAP_only=test_this_IAP_only,
#                                     img_size=img_size,
#                                     mask_class=mask_class,
#                                     seed = seed
#                                 )

# else:
trainset = get_datasets("{}_train".format(dataset_name), 
                        labeling=labeling,
                        task=task,
                        return_filenames=True,
                        balance_classes=balance_classes,
                        test_this_IAP_only=test_this_IAP_only,
                        img_size=img_size,
                        mask_class=mask_class,
                        seed = seed
                    )

valset = get_datasets("{}_val".format(dataset_name), 
                        labeling=labeling,
                        task=task,
                        return_filenames=True,
                        balance_classes=balance_classes,
                        test_this_IAP_only=test_this_IAP_only,
                        img_size=img_size,
                        mask_class=mask_class,
                        seed = seed
                    )

testset_name = "{}_test".format(dataset_name)
if evaluate_harmonization:
    testset_name = harmonized_dataset_name
    print("using harmonized results test set {}".format(testset_name))

testset = get_datasets(testset_name, 
                        labeling=labeling,
                        task=task,
                        return_filenames=True,
                        balance_classes=balance_classes,
                        test_this_IAP_only=test_this_IAP_only,
                        img_size=img_size,
                        mask_class=mask_class,
                        seed = seed
                    )

if save_individual_losses:
    saved_imgfnames = []
    saved_losses = []
    saved_emptymask_flags = []
    saved_labels = []
    bs_eval = 1

if task == "segmentation":
    bs_eval = 1
    print("task is segmentation: changing eval BS to 1 so that avg Dice computed correctly")

print("dataset sizes: train {}, val {}, test {}".format(len(trainset), len(valset), len(testset)))

trainloader = DataLoader(trainset, 
                        batch_size=train_batchsize, 
                        shuffle=True)
valloader = DataLoader(valset, 
                    batch_size=bs_eval)
testloader = DataLoader(testset, 
                    batch_size=bs_eval)

# training augmentations
# custom fat intensity perturbation augmentation
class FatTransform:
    def __init__(self, 
                avg_fat_intensity,
                std_fat_intensity,
                magnitude,
                transform_prob=0.5
                ):
        self.magnitude = magnitude
        self.avg_fat_intensity = avg_fat_intensity
        self.std_fat_intensity = std_fat_intensity
        self.transform_prob = transform_prob


    def __call__(self, img, imgfnames):
        if np.random.rand() <= self.transform_prob:
            fat_mask = []
            #save_image(img, "eg_img_premod.png", normalize=True)
            for j, img_slice in enumerate(img):
                img_slice = np.squeeze(img_slice)
                # extract fat region
                mask_fname = os.path.join(trainset.segmentation_dirs[0], filenames[j])
                # resize to match image using nearest neighbor
                mask_slice = np.array(Image.open(mask_fname).resize(img_slice.shape, Image.NEAREST)).astype(np.uint8)
                # remove overlapping non-fat
                mask_fname_fgtbv = os.path.join(trainset.segmentation_dirs[1], filenames[j])
                # resize to match image using nearest neighbor
                mask_slice_fgtbv = np.array(Image.open(mask_fname_fgtbv).resize(img_slice.shape, Image.NEAREST)).astype(np.uint8)
                final_mask_slice = np.zeros_like(mask_slice)
                final_mask_slice[(mask_slice == 1) & (mask_slice_fgtbv == 0)] = 1
                fat_mask.append(torch.tensor(final_mask_slice).unsqueeze(dim=0))

            fat_mask = torch.stack(fat_mask).bool()

            # perturb fat region intensity
            perturb = self.magnitude * self.std_fat_intensity * np.random.randn()
            img[fat_mask] += perturb

            #save_image(img, "eg_img.png", normalize=True)
            #save_image(fat_mask.float(), "eg_treshmask.png")
       
        return img


if train_with_augmentations:
    if task == 'classification':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=img_size, antialias=True)
        ])
    elif task == 'segmentation':
        # need to get fat intensity statistics for the given domain
        tissue_type = "breast"
        src_domain_radiomics = pd.read_csv(os.path.join(testset.data_dir, "radiomics_domain{}_{}.csv".format(source_domain_label, tissue_type)))
        avg_fat_intensity = src_domain_radiomics['diagnostics_Image-original_Mean'].mean()
        std_fat_intensity = src_domain_radiomics['diagnostics_Image-original_Mean'].std()

        train_transform = FatTransform(
                avg_fat_intensity=avg_fat_intensity,
                std_fat_intensity=std_fat_intensity,
                magnitude=5,
                transform_prob=0.1
                )

    print('\n\n!training with augmentations!\n\n')
    
else:
    train_transform = transforms.Compose([])

# load model
if task == 'segmentation':
    net = model(
        out_classes=1,
        num_encoding_blocks=num_unet_encoding_blocks,
        normalization='batch',
        padding = True,
        )
else:
    net = model()

#print(net)

if task in ['classification', 'regression']:
    # fix first lyr
    make_netinput_onechannel(net, model)
    if task == 'classification':
        num_output_features = num_classes
    elif task == 'regression':
        num_output_features = 1
    else:
        raise NotImplementedError
    change_num_output_features(net, model, num_output_features)

net = net.to(device)
# net = torch.nn.DataParallel(net, device_ids = range(len(device_ids)))

# loss and optim.
if task == 'classification':
    criterion = nn.CrossEntropyLoss()
elif task == 'segmentation':
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = DiceLoss(sigmoid=True)
    def criterion(outputs, targets):
        # flatten targets and outputs except on batch dim
        return criterion1(outputs, targets) + criterion2(outputs, targets)
elif task == 'regression':
    criterion = nn.MSELoss()
else:
    raise NotImplementedError

if train:
    # training
    checkpoint_dir = "saved_models/feature_pred/{}".format(dataset_name)

    # Your code: use an optimizer
    optimizer = torch.optim.Adam(net.parameters(),
                                lr= lr,# default=0.001
                                weight_decay=0.0001     
                            )

    start_epoch = 0

    if task == 'classification' or task == 'segmentation':
        best_val_metric = 0
    elif task == 'regression':
        best_val_metric = np.inf

    if early_stopping:
        epochs_since_best = 0
    for epoch in range(start_epoch, epochs):
        net.train()
        print("Epoch {}:".format(epoch))

        total_examples = 0
        correct_examples = 0

        train_loss = 0

        all_preds = []
        all_gts = []
        all_dices = []

        single_pixel_dist_GE = []
        single_pixel_dist_Siemens = []

        # train for one epoch
        #for batch_idx, (inputs, targets, _) in tqdm(enumerate(trainloader), total=len(trainloader.dataset)//train_batchsize):
        for batch_idx, batch in tqdm(enumerate(trainloader), total=len(trainloader.dataset)//train_batchsize):
            inputs = batch[0].to(device)
            targets = batch[tgt_bidx].to(device).float()
            filenames = batch[2]

            if roi_only:
                roi_masks = batch[trainset.roi_mask_batch_index].to(device)
                inputs[roi_masks == 0] = 0

            if evaluate_domain_shift:
                if "dbc_by_scanner" in dataset_name:
                    domain_labels = trainset.dataset.dataset.get_domain_labels(filenames) 
                else:
                    domain_labels = trainset.get_domain_labels(filenames)
                # 0 = GE, 1 = Siemens
                inputs = inputs[domain_labels == source_domain_label]
                targets = targets[domain_labels == source_domain_label]
                filenames = [f for i, f in enumerate(filenames) if domain_labels[i].item() == source_domain_label]

                if len(targets) == 0:
                    continue

            if use_TR_only is not None:
                assert dataset_name.startswith("dbc")
                if "dbc_by_scanner" in dataset_name:
                    feature_labels = trainset.dataset.dataset.get_feature_labels(filenames) 
                else:
                    feature_labels = trainset.get_feature_labels(filenames)

                if source_domain_label == 0:
                    TRthresh = 5.5
                else:
                    TRthresh = 4
                TRlabels = np.array(feature_labels['TR (Repetition Time)'])
                if use_TR_only == "high":
                    inputs = inputs[TRlabels >= TRthresh]
                    targets = targets[TRlabels >= TRthresh]
                else:
                    inputs = inputs[TRlabels < TRthresh]
                    targets = targets[TRlabels < TRthresh]

                if len(targets) == 0:
                    continue

            # apply transformations
            if task == "segmentation" and train_with_augmentations:
                inputs = train_transform(inputs, filenames)
            else:
                inputs = train_transform(inputs)

            if corrupt_images and not only_corrupt_test_images:
                inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

            # reset gradients
            optimizer.zero_grad()

            # inference
            outputs = net(inputs)
            if task == 'regression':
                outputs = outputs.flatten()
            elif task == 'classification':
                targets = targets.long()
            elif task == 'segmentation':
                # flatten targets and outputs except on batch dim
                targets = targets.view(targets.size(0), -1)
                outputs = outputs.view(outputs.size(0), -1)

            #print(torch.unique(targets, return_counts=True))

            # backprop
            loss = criterion(outputs, targets)
            loss.backward()

            if task == 'classification':
                # Calculate predicted labels
                _, predicted = torch.max(outputs.data, 1)
                # Calculate accuracy
                total_examples += targets.size(0)
                correct_examples += (predicted == targets).sum().item()
                all_preds += predicted.tolist()
                all_gts += targets.tolist()

            elif task == 'segmentation':
                # convert logits to probabilities to predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                all_dices.append(dice_coeff(predicted, targets).item())


            else:
                all_preds += outputs.tolist()
                all_gts += targets.tolist()
                    

            # iterate
            optimizer.step()

            train_loss += loss

        # results
        avg_loss = train_loss / (batch_idx + 1)

        if task == 'classification':
            avg_acc = correct_examples / total_examples
            auc = get_auc_score(all_gts, all_preds)
            print("Training acc: %.4f, Training auc: %.4f" % ( avg_acc, auc))
        elif task == 'segmentation':
            print("N_training examples: {}".format(len(all_dices)))
            avg_dice = np.mean(all_dices) 
            print("Training dice: %.4f" % (avg_dice))
        
        print("Training loss: %.4f" % (avg_loss))

        print(datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        print("Validation...")
        total_examples = 0
        correct_examples = 0

        net.eval()

        # validation
        val_loss = 0
        val_acc = 0

        all_preds = []
        all_gts = []
        all_dices = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(valloader):
                inputs_og = batch[0]
                targets_og = batch[tgt_bidx]
                filenames = batch[2]

                # Copy inputs to device
                inputs = batch[0].to(device)
                targets = batch[tgt_bidx].to(device).float()

                if roi_only:
                    roi_masks = batch[trainset.roi_mask_batch_index].to(device)
                    inputs[roi_masks == 0] = 0

                if evaluate_domain_shift:
                    if "dbc_by_scanner" in dataset_name:
                        domain_labels = valset.dataset.dataset.get_domain_labels(filenames) 
                    else:
                        domain_labels = valset.get_domain_labels(filenames)
                    # 0 = GE, 1 = Siemens
                    inputs = inputs[domain_labels == source_domain_label]
                    targets = targets[domain_labels == source_domain_label]
                    filenames = [f for i, f in enumerate(filenames) if domain_labels[i].item() == source_domain_label]

                    if len(targets) == 0:
                        continue

                if use_TR_only is not None:
                    assert dataset_name.startswith("dbc")
                    if "dbc_by_scanner" in dataset_name:
                        feature_labels = trainset.dataset.dataset.get_feature_labels(filenames) 
                    else:
                        feature_labels = trainset.get_feature_labels(filenames)

                    if source_domain_label == 0:
                        TRthresh = 5.5
                    else:
                        TRthresh = 4.15

                    TRlabels = feature_labels['TR (Repetition Time)']
                    if use_TR_only == "high":
                        inputs = inputs[TRlabels >= TRthresh]
                        targets = targets[TRlabels >= TRthresh]
                    else:
                        inputs = inputs[TRlabels < TRthresh]
                        targets = targets[TRlabels < TRthresh]

                    if len(targets) == 0:
                        continue

                if corrupt_images and not only_corrupt_test_images:
                    inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

                # Generate output from the DNN.
                outputs = net(inputs)
                if task == 'regression':
                    outputs = outputs.flatten()
                elif task == 'classification':
                    targets = targets.long()
                elif task == 'segmentation':
                    # flatten targets and outputs except on batch dim
                    targets = targets.view(targets.size(0), -1)
                    outputs = outputs.view(outputs.size(0), -1)

                # print(targets)
                
                loss = criterion(outputs, targets)            
                if task == 'classification':
                    # Calculate predicted labels
                    _, predicted = torch.max(outputs.data, 1)
                    # Calculate accuracy
                    total_examples += targets.size(0)
                    correct_examples += (predicted == targets).sum().item()
                    all_preds += predicted.tolist()
                    all_gts += targets.tolist()

                elif task == 'segmentation':
                    # convert logits to probabilities to predictions
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    all_dices.append(dice_coeff(predicted, targets).item())

                else:
                    all_preds += outputs.tolist()
                    all_gts += targets.tolist()
                    riginal_gldm_LowGrayLevelEmphasis
                    original_glcm_JointAverage
                    original_glszm_LowGrayLevelZoneEmphasis
                    diagnostics_Image-interpolated_Minimum
                    original_glrlm_ShortRunLowGrayLevelEmphasis
                    original_glrlm_LowGrayLevelRunEmphasis
                    original_glrlm_LongRunLowGrayLevelEmphasis

                val_loss += loss

        avg_loss = val_loss / len(valloader)
        if task == 'classification':
            avg_acc = correct_examples / total_examples
            print("Val acc: %.4f" % (avg_acc))
            auc = get_auc_score(all_gts, all_preds)
            print("Val auc: %.4f" % (auc))
        elif task == 'segmentation':
            avg_dice = np.mean(all_dices) 
            print("Val dice: %.4f" % (avg_dice))
        
        print("Val loss: %.4f" % (avg_loss))


        if early_stopping:
            epochs_since_best += 1
            print("epochs since best: {}".format(epochs_since_best))
            if epochs_since_best >= early_stopping_epochs:
                print("early stopping")
                break
        # Save for checkpoint
        if save_checkpoints:
            check = False
            if task == 'classification':
                if auc > best_val_metric:
                    best_val_metric = auc
                    check = True
            elif task == 'segmentation':
                if avg_dice > best_val_metric:
                    best_val_metric = avg_dice
                    check = True
            elif task == 'regression':
                if avg_loss < best_val_metric:
                    best_val_metric = avg_loss
                    check = True

            if check:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                print("Saving ...")
                state = {'net': net.state_dict(),
                            'epoch': epoch}

                if early_stopping:
                    epochs_since_best = 0

                # delete older checkpoint
                if checkpoint_path_prev:
                    os.remove(checkpoint_path_prev)

                # save new checkpoint
                if task == 'classification':
                    best_val_metric = auc
                elif task == 'segmentation':
                    best_val_metric = avg_dice
                elif task == 'regression':
                    best_val_metric = avg_loss
                checkpoint_path = "{}_{}_{}_{}_{}_{}_best.h5".format(model.__name__,
                    len(trainset), len(valset), best_val_metric, epoch, labeling)

                if corrupt_images:
                    checkpoint_path = checkpoint_path.replace("_best", "_best_bgremoved")
                if evaluate_domain_shift:
                    checkpoint_path = checkpoint_path.replace("_best", "_best_trainedon_domain{}".format(source_domain_label))
                if train_with_augmentations:
                    checkpoint_path = checkpoint_path.replace("_best", "_best_aug")

                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_path)
                torch.save(state, checkpoint_path)

                checkpoint_path_prev = checkpoint_path

        # test
        total_examples = 0
        correct_examples = 0

        if evaluate_domain_shift:
            total_examples_otherdomain = 0
            correct_examples_otherdomain = 0

            all_preds_otherdomain = []
            all_gts_otherdomain = []
            all_dices_otherdomain = []
            all_filenames_otherdomain = []
            all_domainlabels_otherdomain = []

        test_loss = 0
        test_acc = 0

        all_preds = []
        all_gts = []
        all_dices = []
        all_filenames = []
        all_domainlabels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                inputs_og = batch[0]
                targets_og = batch[tgt_bidx]
                filenames_og = batch[2]

                filenames = batch[2]

                # Copy inputs to device
                inputs = batch[0].to(device)
                targets = batch[tgt_bidx].to(device).float()

                if roi_only:
                    roi_masks = batch[trainset.roi_mask_batch_index].to(device)
                    inputs[roi_masks == 0] = 0

                if evaluate_domain_shift:
                    # evaluate in-domain
                    if "dbc_by_scanner" in dataset_name:
                        domain_labels = testset.dataset.dataset.get_domain_labels(filenames) 
                    else:
                        domain_labels = testset.get_domain_labels(filenames)
                    # 0 = GE, 1 = Siemens
                    inputs = inputs[domain_labels == source_domain_label]
                    targets = targets[domain_labels == source_domain_label]
                    filenames = [f for i, f in enumerate(filenames) if domain_labels[i].item() == source_domain_label]
                    all_domainlabels += domain_labels[domain_labels == source_domain_label].tolist()

                    if len(targets) == 0:
                        continue

                if use_TR_only is not None:
                    assert dataset_name.startswith("dbc")
                    if "dbc_by_scanner" in dataset_name:
                        feature_labels = testset.dataset.dataset.get_feature_labels(filenames) 
                    else:
                        feature_labels = testset.get_feature_labels(filenames)

                    if source_domain_label == 0:
                        TRthresh = 5.5
                    else:
                        TRthresh = 4

                    TRlabels = feature_labels['TR (Repetition Time)']

                    if use_TR_only == "high":
                        inputs = inputs[TRlabels >= TRthresh]
                        targets = targets[TRlabels >= TRthresh]
                        filenames = [f for i, f in enumerate(filenames) if TRlabels[i] >= TRthresh]
                        all_domainlabels += domain_labels[TRlabels >= TRthresh].tolist()
                    else:
                        inputs = inputs[TRlabels < TRthresh]
                        targets = targets[TRlabels < TRthresh]
                        filenames = [f for i, f in enumerate(filenames) if TRlabels[i] < TRthresh]
                        all_domainlabels += domain_labels[TRlabels < TRthresh].tolist()

                    if len(targets) == 0:
                        continue

                if corrupt_images:
                    n_viz = 8
                    if batch_idx == 0:
                        save_image(inputs[:n_viz], "eg_prerthreshed_imgs.png")#, normalize=True)
                    inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)
                    if batch_idx == 0 and epoch == 0:
                        plt.figure(figsize=(10,10), dpi=300)

                        #gr = make_grid(inputs[:n_viz], "eg_threshed_imgs.png")
                        gr = make_grid(inputs[:n_viz], nrow=int(np.sqrt(train_batchsize)), pad_value=1)#, normalize=True)
                        gr = (gr - gr.min()) / (gr.max() - gr.min())
                        plt.imshow(gr.permute(1,2,0).cpu())
                        plt.title(targets[:n_viz].cpu().tolist())
                        plt.axis("off")
                        plt.savefig("eg_threshed_imgs.png", bbox_inches="tight")
                        plt.show()

                    if image_corruption_mode == "single_pixel":
                        single_pixel_dist_GE += inputs[:, :, random_pix_ind, random_pix_ind][targets == 0].flatten().cpu().tolist()
                        single_pixel_dist_Siemens += inputs[:, :, random_pix_ind, random_pix_ind][targets == 2].flatten().cpu().tolist()
                    elif image_corruption_mode == "percentile":
                        single_pixel_dist_GE += torch.amax(inputs, dim=(1,2,3))[targets == 0].flatten().cpu().tolist()
                        single_pixel_dist_Siemens += torch.amax(inputs, dim=(1,2,3))[targets == 2].flatten().cpu().tolist()

                #if test_transform is not None:
                #    print("using test transform, DELETE LATER")
                #    inputs = test_transform(inputs)

                # Generate output from the DNN.
                outputs = net(inputs)
                if task == 'regression':
                    outputs = outputs.flatten()
                elif task == 'classification':
                    targets = targets.long()
                elif task == 'segmentation':
                    # flatten targets and outputs except on batch dim
                    targets = targets.view(targets.size(0), -1)
                    outputs = outputs.view(outputs.size(0), -1)

                # print(targets)
                
                loss = criterion(outputs, targets)            
                if task == 'classification':
                    # Calculate predicted labels
                    _, predicted = torch.max(outputs.data, 1)
                    # Calculate accuracy
                    total_examples += targets.size(0)
                    correct_examples += (predicted == targets).sum().item()
                    all_preds += predicted.tolist()
                    all_gts += targets.tolist()

                elif task == 'segmentation':
                    # convert logits to probabilities to predictions
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    all_dices.append(dice_coeff(predicted, targets).item())

                    # use torch imsave to save a batch of images, targets, and predictions
                    if batch_idx == 0 and check:
                        pass
                        #save_image(inputs[:8], "eg_imgs_seg_epoch{}.png".format(epoch), normalize=True)
                        #save_image(targets.view(inputs.shape)[:8], "eg_targets_seg_epoch{}.png".format(epoch))
                        #save_image(predicted.view(inputs.shape)[:8], "eg_preds_seg_epoch{}.png".format(epoch))

                else:
                    all_preds += outputs.tolist()
                    all_gts += targets.tolist()
                    
                all_filenames += filenames

                test_loss += loss

                if evaluate_domain_shift:
                    # also compute accuracy out-of-domain
                    inputs = inputs_og.to(device)[domain_labels == target_domain_label]
                    targets = targets_og.to(device)[domain_labels == target_domain_label].float()
                    filenames = [f for i, f in enumerate(filenames_og) if domain_labels[i] == 1]
                    domain_labels = domain_labels[domain_labels == target_domain_label]

                    if len(targets) == 0:
                        continue

                    if use_TR_only is not None:
                        assert dataset_name.startswith("dbc")
                        if "dbc_by_scanner" in dataset_name:
                            feature_labels = trainset.dataset.dataset.get_feature_labels(filenames)
                        else:
                            feature_labels = trainset.get_feature_labels(filenames)
                        TRlabels = feature_labels['TR (Repetition Time)']
                        if use_TR_only == "high":
                            inputs = inputs[TRlabels >= TRthresh]
                            targets = targets[TRlabels >= TRthresh]
                            filenames = [f for i, f in enumerate(filenames) if TRlabels[i] >= TRthresh]
                            all_domainlabels_otherdomain += domain_labels[TRlabels >= TRthresh].tolist()
                        else:
                            inputs = inputs[TRlabels < TRthresh]
                            targets = targets[TRlabels < TRthresh]
                            filenames = [f for i, f in enumerate(filenames) if TRlabels[i] < TRthresh]
                            all_domainlabels_otherdomain += domain_labels[TRlabels < TRthresh].tolist()

                    if corrupt_images:
                        inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

                    # Generate output from the DNN.
                    outputs = net(inputs)
                    if task == 'regression':
                        outputs = outputs.flatten()
                    elif task == 'classification':
                        targets = targets.long()
                    elif task == 'segmentation':
                        # flatten targets and outputs except on batch dim
                        targets = targets.view(targets.size(0), -1)
                        outputs = outputs.view(outputs.size(0), -1)
                
                    if task == 'classification':
                        # Calculate predicted labels
                        _, predicted = torch.max(outputs.data, 1)
                        # Calculate accuracy
                        total_examples_otherdomain += targets.size(0)
                        correct_examples_otherdomain += (predicted == targets).sum().item()
                        all_preds_otherdomain += predicted.tolist()
                        all_gts_otherdomain += targets.tolist()

                    elif task == 'segmentation':
                        # convert logits to probabilities to predictions
                        predicted = (torch.sigmoid(outputs) > 0.5).float()
                        all_dices_otherdomain.append(dice_coeff(predicted, targets).item())

                        # use torch imsave to save a batch of images, targets, and predictions
                        if batch_idx == 0 and check:
                            save_image(inputs[:8], "eg_imgs_seg_otherdomain_epoch{}.png".format(epoch), normalize=True)
                            save_image(targets.view(inputs.shape)[:8], "eg_targets_seg_otherdomain_epoch{}.png".format(epoch))
                            save_image(predicted.view(inputs.shape)[:8], "eg_preds_seg_otherdomain_epoch{}.png".format(epoch))

                    else:
                        all_preds_otherdomain += outputs.tolist()
                        all_gts_otherdomain += targets.tolist()

                    all_filenames_otherdomain += filenames
                    all_domainlabels_otherdomain += domain_labels.tolist()

        avg_loss = test_loss / len(testloader)
        if task == 'classification':
            avg_acc = correct_examples / total_examples
            print("Test acc: %.4f" % (avg_acc))
            auc = get_auc_score(all_gts, all_preds)
            print("Test auc: %.4f" % (auc))

            if evaluate_domain_shift:
                avg_acc_otherdomain = correct_examples_otherdomain / total_examples_otherdomain
                print("Test acc on other domain: %.4f" % (avg_acc_otherdomain))

                auc_otherdomain = get_auc_score(all_gts_otherdomain, all_preds_otherdomain)
                print("Test auc on other domain: %.4f" % (auc_otherdomain))

        elif task == 'segmentation':
            avg_dice = np.mean(all_dices)
            print("Test dice (N_test = {}): {}".format(len(all_dices), avg_dice))

            if evaluate_domain_shift:
                avg_dice_otherdomain = np.mean(all_dices_otherdomain)
                print("Test dice on other domain: %.4f" % (avg_dice_otherdomain))
            
        elif task == 'regression':
            # errors
            plt.figure()

            # prediction vs. g.t.
            xline = np.linspace(np.min(all_gts), np.max(all_gts), 1000)
            plt.plot(xline, xline, 'k-')
            plt.scatter(all_gts, all_preds, s=10, alpha=0.75)
            plt.xlabel('true value')
            plt.ylabel('predicted value')
            plt.title('{} prediction vs. g.t. \non test set (avg. error: {:.4f})'.format(labeling, avg_loss))
            plt.savefig("pred_vs_gt.png", bbox_inches="tight")
            plt.show()

            if evaluate_domain_shift:
                raise NotImplementedError

        print("Test loss: %.4f" % (avg_loss))

        # look at single pixel dist
        if epoch == 0 and image_corruption_mode == 'single_pixel' and corrupt_images:
            xl = np.max([np.max(single_pixel_dist_GE), np.max(single_pixel_dist_Siemens)]) + 1
            plt.figure(figsize=(3,3))

            # un, cts = np.unique(single_pixel_dist_GE, return_counts=True)
            # plt.plot(un, cts, 'o', label="GE")
            # un, cts = np.unique(single_pixel_dist_Siemens, return_counts=True)
            # plt.plot(un, cts, 'o', label="Siemens")
            nbin = np.linspace(0, xl, 25)
            plt.hist(single_pixel_dist_GE, bins=nbin, alpha=0.5, label="GE", density=True)
            plt.hist(single_pixel_dist_Siemens, bins=nbin, alpha=0.5, label="Siemens", density=True)

            plt.yscale("log")

            #plt.xscale("log")
            plt.xlabel("Intensity")
            plt.ylabel("Density")
            plt.legend()
            plt.xlim(-1,xl)
            plt.title("intensity distribution at pixel ({},{})".format(random_pix_ind, random_pix_ind))
            plt.savefig("visualizations/single_pix_dist.pdf", bbox_inches="tight")
            plt.show()

        # save test set predictions if this is best model so far
        if save_testset_predictions:
            if epochs_since_best == 0:
                if not os.path.exists("testset_predictions"):
                    os.makedirs("testset_predictions")

                if evaluate_domain_shift:
                    # save test set predictions, targets and filenames into .csv
                    testset_predictions_df = pd.DataFrame({'filenames': all_filenames_otherdomain, 'predictions': all_preds_otherdomain, 'targets': all_gts_otherdomain, 'domain_labels': all_domainlabels_otherdomain})
                else:
                    testset_predictions_df = pd.DataFrame({'filenames': all_filenames, 'predictions': all_preds, 'targets': all_gts, 'domain_labels': all_domainlabels})

                testset_predictions_df.to_csv("testset_predictions/{}_{}_{}_{}_{}_{}_testset_predictions.csv".format(model.__name__,
                    len(trainset), len(valset), best_val_metric, epoch, labeling), index=False)


                 


# ## Testing

# In[ ]:


if reload_net:
    state_dict = torch.load(checkpoint_path)['net']
    # remove module from key names
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    net.load_state_dict(state_dict)

    print("loaded checkpoint from {}".format(checkpoint_path))

# plot error distribution vs. slice depth
net.to(device).eval()

if full_test:
# test
    total_examples = 0
    correct_examples = 0

    if evaluate_domain_shift:
        total_examples_otherdomain = 0
        correct_examples_otherdomain = 0

        all_preds_otherdomain = []
        all_gts_otherdomain = []
        all_dices_otherdomain = []
        all_filenames_otherdomain = []
        all_domainlabels_otherdomain = []

    test_loss = 0
    test_acc = 0

    all_preds = []
    all_gts = []
    all_dices = []
    all_filenames = []
    all_domainlabels = []

    # harmonization evaluation
    if eval_harmonization_radiomics:
        # either extract new radiomics or load from file if already exist
        radiomics_csv_fname = os.path.join(data_dirs[testset_name], "radiomics.csv")

        if use_segmentation_radiomics:
            print("NOTICE: using tissue/segmentation-specific radiomics!")
            radiomics_csv_fname = radiomics_csv_fname.replace(".csv", "_seg{}.csv".format(target_mask_batch_index))

        if use_foreground_radiomics:
            print("NOTICE: using foreground-specific radiomics!")
            radiomics_csv_fname = radiomics_csv_fname.replace(".csv", "_foreground.csv")

        #radiomics_csv_fname = 'data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/Siemens/radiomics.csv'
        #radiomics_csv_fname = 'data/dbc/prior_work_1k/mri_data_labeled2D/test/radiomics_domain1_fgt.csv'
    
        #radiomics_csv_fname = 'data/lumbar_mritoct/metric_paper_dataset/harmonized/lumbar_MRI2CT_munit/radiomics.csv'
        #radiomics_csv_fname = 'data/lumbar_mritoct/metric_paper_dataset/MRI/images/test/radiomics.csv'
        #realdata_radiomics_fname ='data/lumbar_mritoct/metric_paper_dataset/CT/images/test/radiomics.csv'

        print("translated image radiomics (will be) saved as {}".format(radiomics_csv_fname))
        if not os.path.exists(radiomics_csv_fname):
            radiomics = []
            radiomics_img_filenames = []

        # load radiomics of real images for later use
        real_radiomics_df = pd.read_csv(realdata_radiomics_fname)
        print("real target domain image radiomics loaded from {}".format(realdata_radiomics_fname))

    if also_harmonize_linearly_w_radiomics:
        print("harmonizing images linearly with radiomics, and saving them")


    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(testloader), total=len(testloader.dataset)//bs_eval):
            inputs_og = batch[0]
            targets_og = batch[tgt_bidx]
            filenames_og = batch[2]

            filenames = batch[2]
            #print(testset.get_domain_labels(filenames))

            # Copy inputs to device
            inputs = batch[0].to(device)
            targets = batch[tgt_bidx].to(device).float()

            if roi_only:
                roi_masks = batch[trainset.roi_mask_batch_index].to(device)
                inputs[roi_masks == 0] = 0

            if evaluate_domain_shift:
                # evaluate in-domain
                if "dbc_by_scanner" in dataset_name:
                    domain_labels = testset.dataset.dataset.get_domain_labels(filenames) 
                else:
                    domain_labels = testset.get_domain_labels(filenames)


                # 0 = GE, 1 = Siemens
                inputs = inputs[domain_labels == source_domain_label]
                targets = targets[domain_labels == source_domain_label]
                filenames = [f for i, f in enumerate(filenames) if domain_labels[i].item() == source_domain_label]
                all_domainlabels += domain_labels[domain_labels == source_domain_label].tolist()
            
            if len(inputs) == 0:
                continue

            if use_TR_only is not None:
                assert dataset_name.startswith("dbc")
                if "dbc_by_scanner" in dataset_name:
                    feature_labels = trainset.dataset.dataset.get_feature_labels(filenames) 
                else:
                    feature_labels = trainset.get_feature_labels(filenames)

                if source_domain_label == 0:
                    TRthresh = 5.5
                else:
                    TRthresh = 4

                TRlabels = feature_labels['TR (Repetition Time)']
                if use_TR_only == "high":
                    inputs = inputs[TRlabels >= TRthresh]
                    targets = targets[TRlabels >= TRthresh]
                    filenames = [f for i, f in enumerate(filenames) if TRlabels[i] >= TRthresh]
                    all_domainlabels += domain_labels[TRlabels >= TRthresh].tolist()
                else:
                    inputs = inputs[TRlabels < TRthresh]
                    targets = targets[TRlabels < TRthresh]
                    filenames = [f for i, f in enumerate(filenames) if TRlabels[i] < TRthresh]
                    all_domainlabels += domain_labels[TRlabels < TRthresh].tolist()

            if corrupt_images:
                n_viz = 8
                if batch_idx == 0:
                    save_image(inputs[:n_viz], "eg_prerthreshed_imgs.png")#, normalize=True)
                inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)
                if batch_idx == 0 and epoch == 0:
                    plt.figure(figsize=(10,10), dpi=300)

                    #gr = make_grid(inputs[:n_viz], "eg_threshed_imgs.png")
                    gr = make_grid(inputs[:n_viz], nrow=int(np.sqrt(train_batchsize)), pad_value=1)#, normalize=True)
                    gr = (gr - gr.min()) / (gr.max() - gr.min())
                    plt.imshow(gr.permute(1,2,0).cpu())
                    plt.title(targets[:n_viz].cpu().tolist())
                    plt.axis("off")
                    plt.savefig("eg_threshed_imgs.png", bbox_inches="tight")
                    plt.show()

                if image_corruption_mode == "single_pixel":
                    single_pixel_dist_GE += inputs[:, :, random_pix_ind, random_pix_ind][targets == 0].flatten().cpu().tolist()
                    single_pixel_dist_Siemens += inputs[:, :, random_pix_ind, random_pix_ind][targets == 2].flatten().cpu().tolist()
                elif image_corruption_mode == "percentile":
                    single_pixel_dist_GE += torch.amax(inputs, dim=(1,2,3))[targets == 0].flatten().cpu().tolist()
                    single_pixel_dist_Siemens += torch.amax(inputs, dim=(1,2,3))[targets == 2].flatten().cpu().tolist()

            #if test_transform is not None:
            #    print("using test transform, DELETE LATER")
            #    inputs = test_transform(inputs)


            if len(targets) == 0:
                continue

            # harmonization eval
            if eval_harmonization_radiomics and not os.path.exists(radiomics_csv_fname):
                for j, image in enumerate(inputs):
                    img_slice = image.cpu().squeeze().numpy()

                    if use_segmentation_radiomics:
                        assert task == 'segmentation'
                        mask_slice = targets[j].cpu().squeeze().numpy().astype(np.uint8)
                        #print(np.unique(mask_slice, return_counts=True))
                        if not 1 in mask_slice:
                            continue
                    elif use_foreground_radiomics:
                        # create segmentation mask with simple thresholding of image
                        thresh_frac = 0.5
                        mask_slice = (img_slice > np.percentile(img_slice, 100*thresh_frac)).astype(np.uint8)
                        plot_egs = False
                        # save image and mask slice for viewing
                        # plot with matplotlib subplots and save
                        if plot_egs:
                            fig, ax = plt.subplots(1, 2, figsize=(10,5))
                            ax[0].imshow(img_slice, cmap='gray')
                            ax[0].set_title("image")
                            ax[1].imshow(mask_slice, cmap='gray')
                            ax[1].set_title("mask")
                            plt.savefig("visualizations/img_and_mask_{}_{}.png".format(batch_idx, j), bbox_inches="tight")
                            plt.show()

                    else:
                        mask_slice = np.ones_like(img_slice).astype(np.uint8)
                        # to prevent bug in pyradiomics https://github.com/AIM-Harvard/pyradiomics/issues/765#issuecomment-1116713745
                        mask_slice[0][0] = 0

                    # compute radiomics
                    features = {}
                    try:
                        features = compute_slice_radiomics(img_slice, mask_slice)
                    except (RuntimeError, ValueError) as e:
                        print(e)
                        print("skipping this slice due to too small ROI...")
                        continue

                    radiomics.append(features)
                    radiomics_img_filenames.append(filenames[j])

            if also_harmonize_linearly_w_radiomics:
                assert "dbc" in dataset_name, "only implemented for dbc dataset"
                assert evaluate_domain_shift
                tissue_types = ["breast", "fgt"]
                tissue_mask_classes = [1, 2]

                # try harmonized both or just one tissue type
                # formatted as:
                #  tissue_types = ["breast", "fgt"]
                # tissue_mask_classes = [1, 2]
                tissue_settings = [
                    [["breast", "fgt"], [1, 2]],
                    [["breast"], [1]],
                    [["fgt"], [2]],
                ]
                for tissue_types, tissue_mask_classes in tissue_settings:
                    harm_levels = np.linspace(0, 1, 5)
                    for harm_level in harm_levels:
                        harmonized_image_dir = 'data/dbc/prior_work_1k/harmonized/by_piecewise_linear_{}/{}to{}/level{}'.format(
                            "-".join(tissue_types), 
                            source_domain_label, 
                            target_domain_label,
                            harm_level
                            )

                        if not os.path.exists(harmonized_image_dir):
                            os.makedirs(harmonized_image_dir)

                        for j, image in enumerate(inputs):
                            img_slice = image.cpu().squeeze().numpy()

                            img_slice_harmonized = img_slice.copy()
                            # for each domain 0 and 1, need radiomics for each tissue type
                            for tissue_idx, tissue_type in enumerate(tissue_types):
                                src_domain_radiomics = pd.read_csv(os.path.join(testset.data_dir, "radiomics_domain{}_{}.csv".format(source_domain_label, tissue_type)))
                                tgt_domain_radiomics = pd.read_csv(os.path.join(testset.data_dir, "radiomics_domain{}_{}.csv".format(target_domain_label, tissue_type)))

                                avg_src_domain_intensity = src_domain_radiomics['diagnostics_Image-original_Mean'].mean()
                                avg_tgt_domain_intensity = tgt_domain_radiomics['diagnostics_Image-original_Mean'].mean()

                                # load segmentation for this tissue type from png to np array
                                # (need to do so instead of using segmentation model target since target is only for one tissue type)
                                mask_fname = os.path.join(testset.segmentation_dirs[tissue_idx], filenames[j])
                                # resize to match image using nearest neighbor
                                mask_slice = np.array(Image.open(mask_fname).resize(img_slice.shape, Image.NEAREST))
                                # only keep class for the tissue type
                                mask_slice = (mask_slice == tissue_mask_classes[tissue_idx]).astype(np.uint8)
                                if not 1 in mask_slice:
                                    continue
                                assert np.unique(mask_slice).tolist() == [0, 1]

                                # interpolate between source and target domain intensity via harm_level parameter

                                print(tissue_types)
                                print(img_slice_harmonized[mask_slice == 1].mean(), img_slice_harmonized[mask_slice == 1].std())
                                print(avg_src_domain_intensity, avg_tgt_domain_intensity)

                                img_slice_harmonized[mask_slice == 1] = img_slice_harmonized[mask_slice == 1] + harm_level * (avg_tgt_domain_intensity - avg_src_domain_intensity) 

                            # save harmonized image as png
                            harmonized_img_fname = os.path.join(harmonized_image_dir, filenames[j])
                            Image.fromarray(img_slice_harmonized).convert("L").save(harmonized_img_fname)
                            print("saved harmonized image to {}".format(harmonized_img_fname))

    
            # Generate output from the DNN.
            outputs = net(inputs)
            if task == 'regression':
                outputs = outputs.flatten()
            elif task == 'classification':
                targets = targets.long()
            elif task == 'segmentation':
                # flatten targets and outputs except on batch dim
                targets = targets.view(targets.size(0), -1)
                outputs = outputs.view(outputs.size(0), -1)

            # print(targets)
            loss = criterion(outputs, targets)            
                

            if save_individual_losses:
                saved_imgfnames.append(filenames[0])

            if task == 'classification':
                # Calculate predicted labels
                _, predicted = torch.max(outputs.data, 1)
                # Calculate accuracy
                total_examples += targets.size(0)
                correct_examples += (predicted == targets).sum().item()
                all_preds += predicted.tolist()
                all_gts += targets.tolist()
                
                if save_individual_losses:
                    #saved_losses.append(loss.item()) # loss 
                    saved_losses.append((predicted == targets).int().item()) # flag for correct or not
                    saved_labels.append(targets.item())

            elif task == 'segmentation':
                # convert logits to probabilities to predictions
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                dice = dice_coeff(predicted, targets).item()
                all_dices.append(dice)

                if save_individual_losses:
                    saved_losses.append(dice)
                    saved_emptymask_flags.append(targets.sum().item() == 0)

            else:
                all_preds += outputs.tolist()
                all_gts += targets.tolist()
            all_filenames += filenames

            test_loss += loss

            if evaluate_domain_shift:
                # also compute accuracy out-of-domain
                inputs = inputs_og.to(device)[domain_labels == target_domain_label]
                targets = targets_og.to(device)[domain_labels == target_domain_label].float()
                filenames = [f for i, f in enumerate(filenames_og) if domain_labels[i] == target_domain_label]
                domain_labels = domain_labels[domain_labels == target_domain_label]

                if len(targets) == 0:
                    continue

                if use_TR_only is not None:
                    assert dataset_name.startswith("dbc")
                    if "dbc_by_scanner" in dataset_name:
                        feature_labels = trainset.dataset.dataset.get_feature_labels(filenames)
                    else:
                        feature_labels = trainset.get_feature_labels(filenames)
                    TRlabels = feature_labels['TR (Repetition Time)']
                    if use_TR_only == "high":
                        inputs = inputs[TRlabels >= TRthresh]
                        targets = targets[TRlabels >= TRthresh]
                        filenames = [f for i, f in enumerate(filenames) if TRlabels[i] >= TRthresh]
                        all_domainlabels_otherdomain += domain_labels[TRlabels >= TRthresh].tolist()
                    else:
                        inputs = inputs[TRlabels < TRthresh]
                        targets = targets[TRlabels < TRthresh]
                        filenames = [f for i, f in enumerate(filenames) if TRlabels[i] < TRthresh]
                        all_domainlabels_otherdomain += domain_labels[TRlabels < TRthresh].tolist()

                if corrupt_images:
                    inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

                # Generate output from the DNN.
                outputs = net(inputs)
                if task == 'regression':
                    outputs = outputs.flatten()
                elif task == 'classification':
                    targets = targets.long()
                elif task == 'segmentation':
                    # flatten targets and outputs except on batch dim
                    targets = targets.view(targets.size(0), -1)
                    outputs = outputs.view(outputs.size(0), -1)
            
                if task == 'classification':
                    # Calculate predicted labels
                    _, predicted = torch.max(outputs.data, 1)
                    # Calculate accuracy
                    total_examples_otherdomain += targets.size(0)
                    correct_examples_otherdomain += (predicted == targets).sum().item()
                    all_preds_otherdomain += predicted.tolist()
                    all_gts_otherdomain += targets.tolist()

                elif task == 'segmentation':
                    # convert logits to probabilities to predictions
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    all_dices_otherdomain.append(dice_coeff(predicted, targets).item())

                else:
                    all_preds_otherdomain += outputs.tolist()
                    all_gts_otherdomain += targets.tolist()

                all_filenames_otherdomain += filenames
                all_domainlabels_otherdomain += domain_labels.tolist()

    if save_individual_losses:
        losses_fname = os.path.join(data_dirs[testset_name], "losses_{}.csv".format(target_mask_batch_index))
        df_dict = {'img_fname': saved_imgfnames, 'loss': saved_losses,}
        if task == "segmentation":
            df_dict['empty_mask_flags'] = saved_emptymask_flags
        elif task == "classification": 
            df_dict['label'] = saved_labels
        losses_df = pd.DataFrame(df_dict)
        losses_df.to_csv(losses_fname, index=False)
        print("saved individual test losses to {}".format(losses_fname))

    # standard task eval
    avg_loss = test_loss / len(testloader)
    if task == 'classification':
        avg_acc = correct_examples / total_examples
        print("Test acc: %.4f" % (avg_acc))
        auc = get_auc_score(all_gts, all_preds)
        print("Test auc: %.4f" % (auc))

        if evaluate_domain_shift:
            avg_acc_otherdomain = correct_examples_otherdomain / total_examples_otherdomain
            print("Test acc on other domain: %.4f" % (avg_acc_otherdomain))

            auc_otherdomain = get_auc_score(all_gts_otherdomain, all_preds_otherdomain)
            print("Test auc on other domain: %.4f" % (auc_otherdomain))

    elif task == 'segmentation':
        avg_dice = np.mean(all_dices)
        print("Test dice (N_test = {}): {}".format(len(all_dices), avg_dice))

        if evaluate_domain_shift:
            avg_dice_otherdomain = np.mean(all_dices_otherdomain)
            print("Test dice on other domain: %.4f" % (avg_dice_otherdomain))
            
    elif task == 'regression':
        # errors
        plt.figure()

        # prediction vs. g.t.
        xline = np.linspace(np.min(all_gts), np.max(all_gts), 1000)
        plt.plot(xline, xline, 'k-')
        plt.scatter(all_gts, all_preds, s=10, alpha=0.75)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        plt.title('{} prediction vs. g.t. \non test set (avg. error: {:.4f})'.format(labeling, avg_loss))
        plt.savefig("pred_vs_gt.png", bbox_inches="tight")
        plt.show()

        if evaluate_domain_shift:
            raise NotImplementedError

    # harmonization eval
    if eval_harmonization_radiomics:
        if os.path.exists(radiomics_csv_fname):
            radiomics_df = pd.read_csv(radiomics_csv_fname)
        else:
            # bring all data together and save
            assert len(radiomics) > 0
            radiomics_df = pd.DataFrame(radiomics)
            radiomics_df.insert(loc=0, column='img_fname', value=radiomics_img_filenames)
            # save df as a csv
            radiomics_df.to_csv(radiomics_csv_fname, index=False)


        if use_segmentation_radiomics:
            print("USING TISSUE-SPECIFIC RADIOMICS: mask class = {}".format(mask_class))

        if use_textural_radiomics_only:
            print("USING TEXTURAL RADIOMICS ONLY.")
            # iterate through column names
            for col in radiomics_df.columns:
                colsplit = col.split('_')
                check = (colsplit[0].split('-')[0] in ["original", "wavelet"]) and (colsplit[1].startswith("gl"))
                #if check:
                #    print(col, colsplit)
                if not check:
                    radiomics_df = radiomics_df.drop(columns=col)
                    real_radiomics_df = real_radiomics_df.drop(columns=col)
        elif use_wavelet_radiomics_only:
            print("USING WAVELET RADIOMICS ONLY.")
            # iterate through column names
            for col in radiomics_df.columns:
                check = col.startswith("wavelet")
                if not check:
                    radiomics_df = radiomics_df.drop(columns=col)
                    real_radiomics_df = real_radiomics_df.drop(columns=col)
        elif use_simple_radiomics_only:
            print("USING SIMPLE RADIOMICS ONLY.")
            # iterate through column names
            for col in radiomics_df.columns:
                check = col == 'diagnostics_Image-original_Mean'
                if not check:
                    radiomics_df = radiomics_df.drop(columns=col)
                    real_radiomics_df = real_radiomics_df.drop(columns=col)

        feats1, feats2 = convert_radiomic_dfs_to_vectors(real_radiomics_df, 
                                                         radiomics_df,
                                                         match_sample_count=True # needed for distance measures
                                                         ) 

        print("radiomic distance measures:")
        # average auc deviation
        auc_devs = []
        for i in range(feats1.shape[1]):
            all_feat_values = np.concatenate([feats1[:, i], feats2[:, i]])
            labels = np.concatenate([np.zeros(feats1.shape[0]), np.ones(feats2.shape[0])])
            auc = roc_auc_score(labels, all_feat_values)
            auc_devs.append(np.abs(auc - 0.5))

        print("avg and median AUC deviation: \n{}\n{}".format(np.mean(auc_devs), np.median(auc_devs)))


        # Frechet distance
        fd = frechet_distance(feats1, feats2)
        print("Frechet Distance: {}".format(fd))

        # Compute the Wasserstein distance between the two datasets
        print(feats1.shape, feats2.shape)
        from geomloss import SamplesLoss

        # Create a SamplesLoss object with the 'sinkhorn' algorithm for efficient computation
        loss = SamplesLoss("sinkhorn")

        # convert np arrays to torch tensors
        feats1 = torch.tensor(feats1).to(device)
        feats2 = torch.tensor(feats2).to(device)

        wasserstein_distance = loss(feats1, feats2)

        print("Wasserstein distance:", wasserstein_distance.item())


        # MMD
        mmd = MMDLoss()(feats1, feats2)
        print("MMD: {}".format(mmd.item()))

# ## grad CAM

# In[ ]:
viz_folder = "visualizations"

if evaluate_fourier:
    # only evaluate for data with the specified label (e.g., cancer)
    tgt_lbl = 2
    # tgt_lbl = None

    all_domain_labels = []

    mean_freqs_x = []
    mean_freqs_y = []

    for batch_idx, batch in tqdm(enumerate(testloader), total=len(testset)//bs_eval, desc="completing forward passes"):
        activations = []
        inputs = batch[0]

        targets = batch[tgt_bidx]
        filenames = batch[2]

        # only do on images with target label
        if tgt_lbl is not None:
            tgt_mask = targets == tgt_lbl
            inputs = inputs[tgt_mask]
            targets = targets[tgt_mask]

            filenames = [f for i, f in enumerate(filenames) if tgt_mask[i].item()]

        # Copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        if "dbc_by_scanner" in dataset_name:
            domain_labels = testset.dataset.dataset.get_domain_labels(filenames) 
        else:
            domain_labels = testset.get_domain_labels(filenames)

        all_domain_labels += domain_labels.tolist()

        if roi_only:
            roi_masks = batch[testset.roi_mask_batch_index].to(device)
            if tgt_lbl is not None:
                roi_masks = roi_masks[tgt_mask]
            inputs[roi_masks == 0] = 0

        if corrupt_images:
            inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)


        # # make signal zero-mean to not have peak at zero
        # inputs = inputs - inputs.mean(dim=(-2,-1), keepdim=True)

        # take 2d fourier transform of images
        # torch version hangs for some reason: inputs = torch.fft.fftn(inputs, dim=(-2,-1))
        for x in inputs:
            spectra = np.fft.fft2(np.squeeze(x.cpu().numpy()))
            mean_freq_x = np.mean(np.abs(spectra), axis=0)
            mean_freq_y = np.mean(np.abs(spectra), axis=1)
            mean_freqs_x += mean_freq_x.tolist()
            mean_freqs_y += mean_freq_y.tolist()


    mean_freqs_x = np.array(mean_freqs_x)
    mean_freqs_y = np.array(mean_freqs_y)
    print("mean mean freq x: {}".format(mean_freqs_x.mean()))
    print("mean mean freq y: {}".format(mean_freqs_y.mean()))
    print("stdev mean freq x: {}".format(mean_freqs_x.std()))
    print("stdev mean freq y: {}".format(mean_freqs_y.std()))
        


if evaluate_feature_space:
    # OPTIONS
    # lyr_name = "relu"

    # resnet18:
    #lyr_name = "layer1" # resnet18
    #lyr_name = "layer4" # resnet18

    # unet:
    #lyr_name = "encoder_encoding_blocks_0_conv1_activation_layer"
    #lyr_name = "encoder_encoding_blocks_0_conv2" # unet
    #lyr_name = "encoder_encoding_blocks_{}_conv2".format(len(net.encoder.encoding_blocks) - 1)  # unet
    #lyr_name = "bottom_block"
    lyr_name = "decoder" # unet

    # only evaluate for data with the specified label (e.g., cancer)
    #tgt_lbl = 0
    tgt_lbl = None

    visualize_domain_cav = True
    use_lossdependent_domain_cav = True
    compare_domain_cav_to_label_cav = False

    viz_method = "lucent"

    loss_cav_grad_type = "loss"
    #loss_cav_grad_type = "prediction"

    if loss_cav_grad_type == "prediction":
        print("using predictions for loss-dependent CAV")

    visualize_feature_perturbs = False
    run_tsne_fresh = False
    compute_dists = False

    # create a list to store the activations
    activations = None
    def get_activation(lyr_name):
        def hook(model, input, output):
            global activations
            activations = output
            if use_lossdependent_domain_cav:
                activations.retain_grad()
        return hook

    # register hook
    lyr = get_layer(net, model, lyr_name)
    hook = lyr.register_forward_hook(get_activation(lyr_name))

    all_activations = []
    all_labels = []
    all_domain_labels = []
    all_filenames = []
    
    if flip_these_domain_labels is not None:
        print("FLIPPING domain labels for images in {}".format(flip_these_domain_labels))

    for batch_idx, batch in tqdm(enumerate(testloader), total=len(testset)//bs_eval, desc="completing forward passes"):
        inputs = batch[0]

        targets = batch[tgt_bidx]
        filenames = batch[2]

        # only do on images with target label
        if tgt_lbl is not None:
            tgt_mask = targets == tgt_lbl
            inputs = inputs[tgt_mask]
            targets = targets[tgt_mask]

            filenames = [f for i, f in enumerate(filenames) if tgt_mask[i].item()]

        # Copy inputs to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        if "dbc_by_scanner" in dataset_name:
            domain_labels = testset.dataset.dataset.get_domain_labels(filenames) 
        else:
            domain_labels = testset.get_domain_labels(filenames)

        if flip_these_domain_labels is not None:
            assert len(np.unique(domain_labels)) <= 2 # only works for binary domains

            ref_fnames = os.listdir(flip_these_domain_labels)
            for i, fname in enumerate(filenames):
                if fname in ref_fnames:
                    print(fname)
                    domain_labels[i] = 1 - domain_labels[i]
        
        all_domain_labels += domain_labels.tolist()
        all_labels += targets.tolist()

        if roi_only:
            roi_masks = batch[testset.roi_mask_batch_index].to(device)
            if tgt_lbl is not None:
                roi_masks = roi_masks[tgt_mask]
            inputs[roi_masks == 0] = 0

        if corrupt_images:
            inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

        # get final conv-layer activations for batch
        outputs = net(inputs)

        all_activations.append(activations.detach().cpu())
        all_filenames += filenames

    all_activations = torch.cat(all_activations, dim=0)

    # compute statistics of activations

    # gram matrices
    grams = torch.einsum('bchw,bdhw->bcd', all_activations, all_activations)
    #grams = all_activations

    if visualize_domain_cav:
        # visualize domain concept activation vector
        all_activations_flattened = all_activations.view(all_activations.shape[0], -1).detach().numpy()

        # subset to N_subset examples
        N_subset = np.min([200, all_activations_flattened.shape[0]])
        subset_indices = np.random.choice(all_activations_flattened.shape[0], N_subset, replace=False)
        all_activations_flattened_subset = all_activations_flattened[subset_indices]
        all_domain_labels_subset = np.array(all_domain_labels)[subset_indices]
        print("subsetted to {} examples for CAV fitting.".format(len(all_domain_labels_subset)))

        # split into train and test using sklearn
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(all_activations_flattened_subset, all_domain_labels_subset, test_size=0.2, random_state=42)

        # first, find the CAV
        from sklearn.linear_model import LogisticRegression

        classifier = LogisticRegression(random_state=0, penalty='l2', max_iter=1000, verbose=1)
        classifier.fit(X_train, y_train)

        print("domain CAV trained accuracy: {}".format(np.mean((y_train == classifier.predict(X_train)).astype(float))))
        print("domain CAV trained AUC: {}".format(get_auc_score(y_train, classifier.predict_proba(X_train)[:,1])))

        print("domain CAV test accuracy: {}".format(np.mean((y_test == classifier.predict(X_test)).astype(float))))
        print("domain CAV test AUC: {}".format(get_auc_score(y_test, classifier.predict_proba(X_test)[:,1])))

        onedirection_cav = classifier.coef_[0]

        # plot sorted average values, sorted by absolute value
        sorted_onedirection_cav = np.abs(onedirection_cav).argsort()[::-1] #.flip([0])

        x = np.arange(len(onedirection_cav))
        y = np.abs(onedirection_cav)[sorted_onedirection_cav]
        plt.figure(figsize=(4,2))
        plt.plot(x, y, 'b-')
        # plt.xscale('log')

        plt.xlabel('sorted training image idx')
        plt.ylabel('avg. $|v_D|$')

        plt.savefig(os.path.join(viz_folder, "domain_cav_acts.png"), bbox_inches="tight")
        plt.show()

        # and plot unsorted
        plt.figure(figsize=(4,2))
        plt.plot(x, onedirection_cav, 'r-')
        # plt.xscale('log')

        plt.xlabel('training image idx')
        plt.ylabel('avg. $|(dL(a)/da)^i|$')

        plt.savefig(os.path.join(viz_folder, "domain_cav_acts_unsorted.png"), bbox_inches="tight")
        plt.show()


        if use_lossdependent_domain_cav:
            # only include features in domain CAV that overlap with features important for loss

            # SETTINGS
            # select which activations (input images) to use in this computation
            # which_activations = "OOD"
            # which_activations = "ID"
            which_activations = "all"
            normalize_importance = True
            choose_from_which_direction = "domain" # "loss"
            only_use_loss_cav = False # don't do any mixing/etc of features from different cavs, just visualize this loss cav

            if only_use_loss_cav:
                print("visualizing loss cav")


            if which_activations == "OOD":
                activation_inds = [a == 1 for a in all_domain_labels]
            elif which_activations == "ID":
                activation_inds = [a == 0 for a in all_domain_labels]
            elif which_activations == "all":
                activation_inds = np.ones(len(all_domain_labels)).astype(bool)
            else:
                raise NotImplementedError

            activation_inds = np.where(activation_inds)[0]

            # create torch subset of test set images and labels
            testset_subset = torch.utils.data.Subset(testset, activation_inds)
            test_subsetloader = DataLoader(testset_subset, 
                    batch_size=bs_eval)

            all_activations_grads = []

            #net = net.to(device).train()

            for batch_idx, batch in tqdm(enumerate(test_subsetloader), total=len(testset_subset)//bs_eval, desc="completing forward passes"):
                inputs = batch[0]

                targets = batch[tgt_bidx]
                filenames = batch[2]

                # only do on images with target label
                if tgt_lbl is not None:
                    tgt_mask = targets == tgt_lbl
                    inputs = inputs[tgt_mask]
                    targets = targets[tgt_mask]

                    filenames = [f for i, f in enumerate(filenames) if tgt_mask[i].item()]

                # Copy inputs to device
                inputs = inputs.to(device)
                targets = targets.float().to(device)

                if roi_only:
                    roi_masks = batch[testset.roi_mask_batch_index].to(device)
                    if tgt_lbl is not None:
                        roi_masks = roi_masks[tgt_mask]
                    inputs[roi_masks == 0] = 0

                if corrupt_images:
                    inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

                # get activations for batch
                outputs = net(inputs)
                if task == 'regression':
                    outputs = outputs.flatten()
                elif task == 'classification':
                    targets = targets.long()
                elif task == 'segmentation':
                    # flatten targets and outputs except on batch dim
                    targets = targets.view(targets.size(0), -1)
                    outputs = outputs.view(outputs.size(0), -1)

                if loss_cav_grad_type == "loss":
                    loss = criterion(outputs, targets)
                    loss.backward() # loss already averaged over batch
                elif loss_cav_grad_type == "prediction":
                    outputs.mean().backward() # average over batch, and classes/segmentation pixels
                else:
                    raise NotImplementedError

                # get gradient of loss wrt activations
                activation_grads = activations.grad
                #print(activation_grads.shape)
                all_activations_grads.append(activation_grads.cpu())

            all_activations_grads = torch.cat(all_activations_grads, dim=0)

            # compute TCAV concept sensitivity
            #tcav_sensitive = torch.einsum('bn,n->b', all_activations_grads.view(all_activations_grads.shape[0], -1).to(device).float(), torch.from_numpy(onedirection_cav).to(device).float())
            tcav_sensitive = torch.Tensor(np.einsum('bn,n->b', all_activations_grads.view(all_activations_grads.shape[0], -1).numpy(), onedirection_cav)).to(device)

            #print(all_activations_grads.shape, all_activations_grads, tcav_sensitive)
            # count fraction that is positive
            tcav_sensitivity = (tcav_sensitive > 0).float().mean()
            print("TCAV sensitivity: {}".format(tcav_sensitivity))

            # get average absolute value of gradient of loss wrt activations
            take_abs_grad = False
            preabs_avg_grads = all_activations_grads.mean(dim=0)
            if take_abs_grad:
                avg_grads = torch.abs(preabs_avg_grads)
            else:
                avg_grads = preabs_avg_grads
            avg_grads = avg_grads.view(-1)

            # plot sorted average gradient values
            sorted_avg_grads = avg_grads.argsort().flip([0])

            x = np.arange(len(avg_grads))
            y = avg_grads[sorted_avg_grads]
            plt.figure(figsize=(4,2))
            plt.plot(x, y, 'b-')
            # plt.xscale('log')

            plt.xlabel('sorted training image idx')
            plt.ylabel('avg. $|(dL(a)/da)^i|$')

            plt.savefig(os.path.join(viz_folder, "loss_cav_acts.png"), bbox_inches="tight")
            plt.show()

            # and plot unsorted
            plt.figure(figsize=(4,2))
            plt.plot(x, avg_grads, 'r-')
            # plt.xscale('log')

            plt.xlabel('training image idx')
            plt.ylabel('avg. $|(dL(a)/da)^i|$')

            plt.savefig(os.path.join(viz_folder, "loss_cav_acts_unsorted.png"), bbox_inches="tight")
            plt.show()


            loss_cav = avg_grads.numpy()

            # convert to unit vector
            loss_cav = loss_cav / np.linalg.norm(loss_cav)

            # compute similarity between label and domain CAVs
            cossim = cosine_similarity(torch.from_numpy(onedirection_cav).to(device), torch.from_numpy(loss_cav).to(device))
            print("cossim between |vL| and vD CAVs: {}".format(cossim))

            if only_use_loss_cav:
                onedirection_cav = loss_cav
            else:
                # only keep features in domain CAV that are also important for loss (or vice versa)
                #thresh = 1e-4
                # feature_importance = np.abs(onedirection_cav * loss_cav)
                feature_importance = onedirection_cav * loss_cav

                if choose_from_which_direction == "domain":
                    main_feature_direction = onedirection_cav
                elif choose_from_which_direction == "loss":
                    main_feature_direction = loss_cav
                else:
                    raise NotImplementedError

                # print(np.min(feature_importance), np.max(feature_importance))

                thresh = np.percentile(feature_importance, 99)
                # thresh = 0.1

                preprune_onedirection_cav = onedirection_cav
                print("CAV nonzero feature before pruning: {}".format(np.count_nonzero(onedirection_cav)))
                onedirection_cav = np.where(feature_importance >= thresh, main_feature_direction, 0)
                print("CAV nonzero feature after pruning: {}".format(np.count_nonzero(onedirection_cav)))

                preabs_loss_cav = preabs_avg_grads.view(-1).numpy()
                preabs_loss_cav = preabs_loss_cav / np.linalg.norm(preabs_loss_cav)
                cossim = cosine_similarity(torch.from_numpy(preprune_onedirection_cav).to(device), torch.from_numpy(preabs_loss_cav).to(device))
                print("cossim between vL and vD CAVs: {}".format(cossim))

                cossim = cosine_similarity(torch.from_numpy(onedirection_cav).to(device), torch.from_numpy(preprune_onedirection_cav).to(device))
                print("cossim between vD' and vD CAVs after pruning domain CAV: {}".format(cossim))

                cossim = cosine_similarity(torch.from_numpy(onedirection_cav).to(device), torch.from_numpy(loss_cav).to(device))
                print("cossim between vD' and |vL| CAVs after pruning domain CAV: {}".format(cossim))

                cossim = cosine_similarity(torch.from_numpy(onedirection_cav).to(device), torch.from_numpy(preabs_loss_cav).to(device))
                print("cossim between vD' and vL CAVs after pruning domain CAV: {}".format(cossim))


        cav = np.array([-onedirection_cav, onedirection_cav])
        # captures cav in both directions

        if compare_domain_cav_to_label_cav:
            all_labels_subset = np.array(all_labels)[subset_indices]
            X_train, X_test, y_train, y_test = train_test_split(all_activations_flattened_subset, all_labels_subset, test_size=0.2, random_state=42)

            classifier = LogisticRegression(random_state=0, penalty='l2', max_iter=2000, verbose=1)
            classifier.fit(X_train, y_train)

            print("label CAV trained accuracy: {}".format(np.mean((y_train == classifier.predict(X_train)).astype(float))))
            print("label CAV trained AUC: {}".format(get_auc_score(y_train, classifier.predict_proba(X_train)[:,1])))

            print("label CAV test accuracy: {}".format(np.mean((y_test == classifier.predict(X_test)).astype(float))))
            print("label CAV test AUC: {}".format(get_auc_score(y_test, classifier.predict_proba(X_test)[:,1])))

            label_cav = torch.from_numpy(classifier.coef_[0]).to(device)

            # compute similarity between label and domain CAVs
            cossim = cosine_similarity(torch.from_numpy(cav[0]).to(device), label_cav)
            print("cossim between label and domain CAVs: {}".format(cossim))



        # load model
        if task == 'segmentation':
            net = model(
                out_classes=1,
                num_encoding_blocks=num_unet_encoding_blocks,
                normalization='batch',
                padding = True,
                )
        else:
            net = model()
        if task in ['classification', 'regression']:
            # fix first lyr
            make_netinput_onechannel(net, model)
            if task == 'classification':
                num_output_features = num_classes
            elif task == 'regression':
                num_output_features = 1
            else:
                raise NotImplementedError
            change_num_output_features(net, model, num_output_features)

        net.load_state_dict(state_dict)
        net = net.to(device).eval()
        

        if viz_method == "lucent":
            # visualize with lucent
            @objectives.wrap_objective()
            def dot_cav(layer, cav, opt_sim=1., cossim_pow=0):
                def inner(T):
                    dot = (cav.T * T(layer).view(-1)).sum()
                    mag = torch.sqrt(torch.sum(T(layer)**2))
                    cossim = dot / (1e-6 + mag)

                    ret = -dot * cossim ** cossim_pow
                    return ret
                return inner
            lucent_iters = 2000

        for which_cav_direction in [0, 1]:
            if viz_method == "lucent":
                param_f = lambda: param.pixel_image(shape=(1, 1, img_size, img_size))

                obj = dot_cav(lyr_name, torch.from_numpy(cav[which_cav_direction]).to(device))
                out = render.render_vis(net, obj, param_f, verbose=True, show_image=False, thresholds=(lucent_iters, ), 
                        transforms=[], preprocess=False) # <- make work with 1 channel imgs

            # save out array as an image
            out = np.squeeze(out[0][0])
            out = (out - out.min()) / (out.max() - out.min())

            # save with PIL
            out_img = Image.fromarray(np.uint8(out * 255))
            img_savepath = os.path.join(viz_folder, "cav{}_{}_{}.png".format(which_cav_direction, dataset_name, lyr_name))
            out_img.save(img_savepath)
            print("saved CAV image to {}".format(img_savepath))

            if visualize_feature_perturbs: 
                # start with a random image from one domain, perturb it's features in the direction of the CAV towards the other domain, and visualize the features
                img_idx = all_domain_labels.index(1 - which_cav_direction)
                img_fname = all_filenames[img_idx]
                print("starting image: {}".format(img_fname))

                act_0 = all_activations[img_idx].view(-1)
                for step in [0, 1e2, 1e3, 1e4, 1e5]: #[0, 0.01, 0.1, 0.5, 1., 1.5, 2, 5, 10]:
                #for step in [0, 0.2, 0.4, 0.6, 0.8, 1]: #[0, 0.01, 0.1, 0.5, 1., 1.5, 2, 5, 10]:

                    tgt_act = act_0 + step * torch.from_numpy(cav[which_cav_direction])
                    #tgt_act = torch.from_numpy(cav[which_cav_direction])

                    if viz_method == "lucent":
                        # visualize image that results in this activation
                        @objectives.wrap_objective()
                        def match_act(layer, tgt_activation):
                            def mse(T):
                                return torch.mean((tgt_activation - T(layer).view(-1))**2)
                            return mse
                        #def dot_cav(layer, cav, optimal_similarity=step):
                        #    def inner(T):
                        #        dot = (cav.T * T(layer).view(-1)).sum()
                        #        mag = torch.sqrt(torch.sum(T(layer)**2))
                        #        ret = - dot / (1e-6 + mag)
                        #        if optimal_similarity != 1:
                        #            # optimize vectors to have some similarity
                        #            ret = np.abs(ret - (-optimal_similarity))
                        #        return ret
                        #    return inner

                        param_f = lambda: param.pixel_image(shape=(1, 1, img_size, img_size))

                        obj = match_act(lyr_name, tgt_act.to(device))
                        out = render.render_vis(net, obj, param_f, verbose=True, show_image=False, thresholds=(lucent_iters, ), 
                                transforms=[], preprocess=False) # <- make work with 1 channel imgs

                    # save out array as an image
                    out = np.squeeze(out[0][0])
                    out = (out - out.min()) / (out.max() - out.min())

                    # save with PIL
                    out_img = Image.fromarray(np.uint8(out * 255))
                    img_savepath = os.path.join(viz_folder, "cav{}_{}_{}_+{}.png".format(which_cav_direction, dataset_name, lyr_name, step))
                    out_img.save(img_savepath)
                    print("saved perturbed image to {}".format(img_savepath))


                do_cav_fourier = False
                if do_cav_fourier:
                    # load image from file with PIL to perform FFT on (testing only)
                    #out = np.array(Image.open('data/dbc/png_subset/sorted_by_scanner/GE/1-090-083.png').convert('L').resize((img_size, img_size)))
                    # out = np.array(Image.open('data/dbc/png_subset/sorted_by_scanner/Siemens/1-094-420.png').convert('L').resize((img_size, img_size)))


                    # # make signal zero-mean to not have peak at zero
                    # out = out - np.mean(out)
                    # analyze image in fourier space
                    out_fourier = np.fft.fftshift(np.fft.fft2(out))
                    plt.figure(figsize=(4, 4), dpi=80)
                    plt.imshow(np.log(abs(out_fourier)))#, cmap='hot')
                    # plt.imshow(abs(out_fourier))#, cmap='hot')
                    plt.title("Spatial log power spectrum\n of CAV image")
                    plt.savefig(os.path.join(viz_folder, "cav{}_{}_fourier.png".format(round(lamb_interp, 2), lyr_name)), bbox_inches="tight")

                    # plot zoomed in images
                    plt.figure(figsize=(4, 4), dpi=80)
                    plt.imshow(np.log(abs(out_fourier)))#, cmap='hot')
                    plt.title("Spatial log power spectrum\n of CAV image (zoomed)")
                    plt.xlim(out.shape[0]//2 - 10, out.shape[0]//2 + 10)
                    plt.ylim(out.shape[1]//2 - 10, out.shape[1]//2 + 10)
                    plt.savefig(os.path.join(viz_folder, "cav{}_{}_fourier_zoomed.png".format(round(lamb_interp, 2), lyr_name)), bbox_inches="tight")

    hook.remove()

    if run_tsne_fresh:
        # t-sne to visualize
        from sklearn.manifold import TSNE
        grams_emb = TSNE(
            n_components=2, perplexity=50, n_iter=1000, verbose=True
            ).fit_transform(
                grams.view(grams.shape[0], -1).numpy()
                            )

        # plot with seaborn with domain labels
        plt.figure(figsize=(6,6), dpi=300)
        g = sns.scatterplot(x=grams_emb[:,0], y=grams_emb[:,1], hue=all_domain_labels, palette="tab10")
        plt.title("Gram matrices of feature maps\n in layer {}".format(lyr_name))

        legend_handles, _= g.get_legend_handles_labels()
        g.legend(legend_handles, ['GE', 'Siemens'], 
            bbox_to_anchor=(1,1), 
            title="Domain")
        
        plt.savefig(os.path.join(viz_folder, "grams_emb.pdf"), bbox_inches="tight")


    if compute_dists:
        # compute gram matrix FID between GE and Siemens
        # only take diagonal of grams since otherwise too large

        all_domain_labels = np.array(all_domain_labels)

        grams = torch.diagonal(grams, offset=0, dim1=1, dim2=2)
        grams = grams.view(grams.shape[0], -1).numpy()
        print(grams.shape)
        print(all_activations.shape)
        grams1 = grams[all_domain_labels == 0]
        grams2 = grams[all_domain_labels == 1]
        mean1 = np.mean(grams1, axis=0)
        sigma1 = np.cov(grams1, rowvar=False)
        mean2 = np.mean(grams2, axis=0)
        sigma2 = np.cov(grams2, rowvar=False)

        fid = fid_score.calculate_frechet_distance(mean1, sigma1, mean2, sigma2)
        print("Frechet Distance between GE and Siemens gram matrices diagonals: {}".format(fid))

        # compute activation FID between GE and Siemens
        all_activations = all_activations.view(all_activations.shape[0], -1).numpy()
        all_activations1 = all_activations[all_domain_labels == 0]
        all_activations2 = all_activations[all_domain_labels == 1]
        mean1 = np.mean(all_activations1, axis=0)
        sigma1 = np.cov(all_activations1, rowvar=False)
        mean2 = np.mean(all_activations2, axis=0)
        sigma2 = np.cov(all_activations2, rowvar=False)

        fid = fid_score.calculate_frechet_distance(mean1, sigma1, mean2, sigma2)
        print("Frechet Distance between GE and Siemens: {}".format(fid)) 

if do_attributions:
    if task != 'classification':
        raise NotImplementedError

    run_trak_fresh = True

    # need non-shuffled trainloader for TRAK
    trainloader_trak = DataLoader(trainset, 
                        batch_size=train_batchsize)

    assert fix_seed and seed == 1338
    from trak import TRAKer

    # setup
    trak_ckpt_dir = "saved_models/feature_pred/dbc_priorwork1k/fortrak_breastonly"
    ckpt_files = os.listdir(trak_ckpt_dir)
    ckpt_files = [os.path.join(trak_ckpt_dir, ckpt_file) for ckpt_file in ckpt_files]
    ckpts = [torch.load(ckpt, map_location='cpu')['net'] for ckpt in ckpt_files]
    ckpts = [{k.replace("module.", ""): v for k, v in ckpt.items()} for ckpt in ckpts]

    print(len(ckpts))

    traker = TRAKer(model=net,    # model to be explained
                task='image_classification',
                train_set_size=len(trainset),
                use_half_precision=True
                )
    if run_trak_fresh: 
        # compute features of training data
        featurize_fresh = True
        if featurize_fresh:
            for model_id, ckpt in enumerate(tqdm(ckpts)):
                # TRAKer loads the provided checkpoint and also associates
                # the provided (unique) model_id with the checkpoint.
                traker.load_checkpoint(ckpt, model_id=model_id)

                for batch in tqdm(trainloader_trak, total=len(trainloader_trak.dataset)//train_batchsize):
                    if roi_only:
                        roi_masks = batch[trainset.roi_mask_batch_index].to(device)

                    batch = [x.cuda() for x in batch[:2]]

                    if roi_only:
                        batch[0][roi_masks == 0] = 0

                    if corrupt_images and not only_corrupt_test_images:
                        batch[0] = apply_image_corruption(batch[0], image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

            # reset gradients
                    # TRAKer computes features corresponding to the batch of examples,
                    # using the checkpoint loaded above.
                    traker.featurize(batch=batch, num_samples=batch[0].shape[0])

            # Tells TRAKer that we've given it all the information, at which point
            # TRAKer does some post-processing to get ready for the next step
            # (scoring target examples).
            traker.finalize_features()

        # compute trak scores for test examples
        for model_id, ckpt in enumerate(tqdm(ckpts)):
            traker.start_scoring_checkpoint(exp_name='quickstart',
                                            checkpoint=ckpt,
                                            model_id=model_id,
                                            num_targets=len(testset))
            for batch in testloader:
                if roi_only:
                    roi_masks = batch[trainset.roi_mask_batch_index].to(device)

                batch = [x.cuda() for x in batch[:2]]

                if roi_only:
                    batch[0][roi_masks == 0] = 0

                if corrupt_images and not only_corrupt_test_images:
                    batch[0] = apply_image_corruption(batch[0], image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

                traker.score(batch=batch, num_samples=batch[0].shape[0])

        scores = traker.finalize_scores(exp_name='quickstart')

    else:
        from numpy.lib.format import open_memmap
        scores = open_memmap('./trak_results/scores/quickstart.mmap')  

    # visualize specific examples
    for i in [7, 21, 22, 23, 24, 25, 26]:
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3), dpi=300)
        fig.suptitle('Top scoring TRAK images from the train set')

        test_img = testset[i][0]

        if roi_only:
            roi_mask = testset[i][trainset.roi_mask_batch_index]
            test_img[roi_mask == 0] = 0

        if corrupt_images:
            test_img = apply_image_corruption(test_img, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

        axs[0].imshow(torch.squeeze(test_img), cmap="gray")
        if "dbc_by_scanner" in dataset_name:
            testimg_domainlabel = testset.dataset.dataset.get_domain_labels([testset[i][2]]).item()
        else:
            testimg_domainlabel = testset.get_domain_labels([testset[i][2]]).item()
        axs[0].set_title('Test image\n domain = {}'.format(testimg_domainlabel))
        
        axs[0].axis('off')
        axs[1].axis('off')
        
        top_trak_scorers = scores[:, i].argsort()[-5:][::-1]
        for ii, train_im_ind in enumerate(top_trak_scorers):
            train_img = trainset[train_im_ind][0]

            if roi_only:
                roi_mask = trainset[train_im_ind][trainset.roi_mask_batch_index]
                train_img[roi_mask == 0] = 0

            if corrupt_images and not only_corrupt_test_images:
                train_img = apply_image_corruption(train_img, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

            axs[ii + 2].imshow(torch.squeeze(train_img), cmap="gray")

            if "dbc_by_scanner" in dataset_name:
                trimg_domainlabel = trainset.dataset.dataset.get_domain_labels([trainset[train_im_ind][2]]).item()
            else:
                trimg_domainlabel = trainset.get_domain_labels([trainset[train_im_ind][2]]).item()
            axs[ii + 2].set_title('domain = {}'.format(trimg_domainlabel))
            axs[ii + 2].axis('off')

        plt.savefig(os.path.join(viz_folder, "trak_{}.png".format(i)))
        fig.show()

    # analyze average importance of training samples over the entire test set

    # plot score distribution
    avg_tr_scores = np.mean(scores, axis=1)
    top_trak_scorers = avg_tr_scores.argsort()[::-1]

    x = np.arange(len(avg_tr_scores))
    y = avg_tr_scores[top_trak_scorers]
    plt.figure(figsize=(4,2))
    plt.plot(x, y, 'b-')
    # plt.xscale('log')

    plt.xlabel('sorted training image idx')
    plt.ylabel('avg. attrib. score $\tau_c(x_\mathrm{tr})$')
    plt.grid()

    plt.savefig(os.path.join(viz_folder, "trak_avg_scores.png"), bbox_inches="tight")
    plt.show()

    # save filenames of top scoring training images
    fnames = []
    for train_im_ind in tqdm(top_trak_scorers):
        fnames.append(trainset[train_im_ind][2])

    # save filenames
    with open(os.path.join(viz_folder, "trak_topavg_fnames.txt"), "w") as f:
        f.write("\n".join(fnames))

    # plot highest attrib images on average
    num_plt = 14
    fig, axs = plt.subplots(ncols=num_plt//2, nrows=2, figsize=(num_plt, 6), dpi=300)
    for ii, train_im_ind in enumerate(top_trak_scorers[:num_plt]):
        j = ii % 2
        k = ii // 2

        train_img = trainset[train_im_ind][0]

        if roi_only:
            roi_mask = trainset[train_im_ind][trainset.roi_mask_batch_index]
            train_img[roi_mask == 0] = 0

        if corrupt_images and not only_corrupt_test_images:
            train_img = apply_image_corruption(train_img, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

        axs[j][k].imshow(torch.squeeze(train_img), cmap="gray")
        if "dbc_by_scanner" in dataset_name:
            trimg_domainlabel = trainset.dataset.dataset.get_domain_labels([trainset[train_im_ind][2]]).item()
        else:
            trimg_domainlabel = trainset.get_domain_labels([trainset[train_im_ind][2]]).item()
        axs[j][k].set_title('domain = {}'.format(trimg_domainlabel))
        axs[j][k].axis('off')

    plt.savefig(os.path.join(viz_folder, "trak_topavg.png"), bbox_inches="tight")
    plt.show()

    # do the same, but only for test images from one domain
    for domain_label in [0, 1]:
        train_fnames = [x[2] for x in trainset]
        if "dbc_by_scanner" in dataset_name:
            avg_tr_scores_thisdomain = avg_tr_scores[trainset.dataset.dataset.get_domain_labels(train_fnames) == domain_label] 
        else:
            avg_tr_scores_thisdomain = avg_tr_scores[trainset.get_domain_labels(train_fnames) == domain_label] 
        top_trak_scorers = avg_tr_scores_thisdomain.argsort()[::-1] 

        fig, axs = plt.subplots(ncols=num_plt//2, nrows=2,  figsize=(num_plt, 6), dpi=300)
        for ii, train_im_ind in enumerate(top_trak_scorers[:num_plt]):
            j = ii % 2
            k = ii // 2

            train_img = trainset[train_im_ind][0]

            if roi_only:
                roi_mask = trainset[train_im_ind][trainset.roi_mask_batch_index]
                train_img[roi_mask == 0] = 0

            if corrupt_images and not only_corrupt_test_images:
                train_img = apply_image_corruption(train_img, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

            axs[j][k].imshow(torch.squeeze(train_img), cmap="gray")

            if "dbc_by_scanner" in dataset_name:
                trimg_domainlabel = trainset.dataset.dataset.get_domain_labels([trainset[train_im_ind][2]]).item()
            else:
                trimg_domainlabel = trainset.get_domain_labels([trainset[train_im_ind][2]]).item()
            axs[j][k].set_title('domain = {}'.format(trimg_domainlabel))
            axs[j][k].axis('off')

        plt.savefig(os.path.join(viz_folder, "trak_topavg_testdomain{}.png".format(domain_label)), bbox_inches="tight")
        plt.show()



if do_gradcam:
    print('running gradCAM')

    tgt_lbl = None
    plot_tumor_bboxes = False
    
    # Construct the CAM object once, and then re-use it on many images:
    if task == "classification":
        try:
            target_layers = [net.module.layer4[-1]]
        except AttributeError:
            target_layers = [net.layer4[-1]]
    else:
        raise NotImplementedError
    cam = GradCAM(model=net, target_layers=target_layers) #, use_cuda=use_cuda)

    num_viz = 10
    num_vizzed = 0

    gts = []
    preds = []

    for batch_idx, batch in enumerate(testloader):
        # Copy inputs to device
        inputs = batch[0].to(device)
        targets = batch[tgt_bidx].to(device).float()
        filenames = batch[2]

        if roi_only:
            roi_masks = batch[trainset.roi_mask_batch_index].to(device)
            inputs[roi_masks == 0] = 0

        if corrupt_images:
            inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)
        # Generate output from the DNN.
        outputs = net(inputs)
        if task == 'classification':
            # Calculate predicted labels
            _, predicted = torch.max(outputs.data, 1)

        elif task == 'segmentation':
            # convert logits to probabilities to predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()

        gts += targets.tolist()
        preds += predicted.tolist()

        if "dbc_by_scanner" in dataset_name:
            domain_labels = testset.dataset.dataset.get_domain_labels(filenames)
        else:
            domain_labels = testset.get_domain_labels(filenames)

        #print(targets.unique(return_counts=True))

        for i in range(inputs.shape[0]):
    #        if i % (inputs.shape[0] // num_viz) == 0:
            one_input = inputs[i].unsqueeze(0)

            # print(targets[i].item(), tgt_lbl)
        # print(targets[i].item())
            if tgt_lbl is not None and targets[i].item() != tgt_lbl:
                continue
            
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=one_input)
            fig, axs = plt.subplots(2, 1, figsize=(3,6))
            # remove axes
            for ax in axs:
                ax.axis('off')
            axs[0].imshow(one_input.squeeze().cpu().numpy(), cmap='gray') #.transpose(1,2,0))
            axs[1].imshow(one_input.squeeze().cpu().numpy()) #.transpose(1,2,0))
            axs[1].imshow(grayscale_cam[0], alpha=0.4, cmap='jet')
            #plt.title('{}\n g.t. label: {} pred: {}\n domain = {}'.format(labeling, targets[i].item(), predicted[i].item(), domain_labels[i]))

            if plot_tumor_bboxes:
                # get tumor bbox
                bbox = trainset.get_tumor_bboxes([filenames[i]])[0]
                # print(bbox)
                rect = patches.Rectangle((bbox[0], bbox[3] + bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], linewidth=4, edgecolor='r', facecolor='none')
                axs[1].add_patch(rect)

            plt.savefig(os.path.join(viz_folder, "gradcam_{}_{}_domain{}.png".format(labeling, i, domain_labels[i])))
            plt.show()
            num_vizzed += 1
            print("plotted {}".format(num_vizzed))

            if num_vizzed >= num_viz:
                break

        break

    print("auc = {}".format(get_auc_score(gts, preds)))


# do adversarial attack
attack_type = "FGSM"
atk_eps = 16
num_viz = 5
vizzed = False
tgt_lbl = None # 

save_attacked_images = True
atk_save_dir = os.path.join(viz_folder, "attack_experiment")
if save_attacked_images and not os.path.exists(atk_save_dir):
    os.makedirs(atk_save_dir)
    print("saving attacked images from domain 1 and non-attacked images from domain 0 to {}".format(atk_save_dir))

if adv_attack:
    # setup downstream task model to see if attacks can harmonize
    net_cancer = resnet18()
    # fix first lyr
    make_netinput_onechannel(net_cancer, resnet18)
    num_output_features = 2
    change_num_output_features(net_cancer, resnet18, num_output_features)

    net_cancer = net_cancer.to(device)
    # net_cancer = torch.nn.DataParallel(net_cancer, device_ids = range(len(device_ids)))
    # checkpoint_path_cancer = "saved_models/feature_pred/dbc_by_scanner/resnet18_9965_2077_0.9951853635050554_0_feature: Manufacturer_best.h5"
    # net_cancer.load_state_dict(torch.load(checkpoint_path_cancer)['net'])
    print("TODO: load cancer net checkpoint")
    net_cancer.eval()

    total_examples_cancer = 0
    correct_examples_cancer = 0
    correct_examples_cancer_atked = 0

    targs = []
    preds = []

    # for AUC eval
    targs_cancer = []
    preds_cancer = []
    preds_cancer_atked = []
    preds_cancer2 = []
    preds_cancer_atked2 = []
    #
    eg_atks = torch.empty((3, num_viz, 1, img_size, img_size))

    # get adv accuracy
    total_examples = 0
    correct_examples = 0
    test_loss = 0

    #correct_examples_othermodel = 0
    for batch_idx, batch in tqdm(enumerate(testloader)):

        # Copy inputs to device
        inputs = batch[0].to(device)
        targets = batch[tgt_bidx].to(device).float()
        filenames = batch[2]

        # only do attack on images with target label
        if tgt_lbl is not None:
            tgt_mask = targets == tgt_lbl
            inputs = inputs[tgt_mask]
            targets = targets[tgt_mask]

            filenames = [f for i, f in enumerate(filenames) if tgt_mask[i].item()]

        if roi_only:
            roi_masks = batch[trainset.roi_mask_batch_index].to(device)
            if tgt_lbl is not None:
                roi_masks = roi_masks[tgt_mask]
            inputs[roi_masks == 0] = 0

        if corrupt_images:
            inputs = apply_image_corruption(inputs, image_corruption_mode, random_pix_ind, background_removal_frac, noise_level)

        inputs.requires_grad = True

        # generate attack
        adv_images = generate_attack(attack_type, net, criterion, inputs, targets, atk_eps).detach()
        outputs = net(adv_images).detach()
        loss = criterion(outputs, targets)

        _, predicted = torch.max(outputs.data, 1)
        targs += targets.cpu().tolist()
        preds += predicted.cpu().tolist()

        # save attacked images from domain A, and non attacked images from domain B
        if save_attacked_images:
            assert tgt_lbl is None, "need to save images from both domains"
            for j in range(inputs.shape[0]):
                if "dbc_by_scanner" in dataset_name:
                    domain = testset.dataset.dataset.get_domain_labels([filenames[j]])[0]
                else:
                    domain = testset.get_domain_labels([filenames[j]])[0]
                if domain == 0:
                    save_image(inputs[j], os.path.join(atk_save_dir, filenames[j].split(".")[0] + ".png"))
                else:
                    save_image(adv_images[j], os.path.join(atk_save_dir, filenames[j].split(".")[0] + ".png"))

        # evaluate cancer detection model before and after attack
        targets_cancer = testset.get_cancer_labels(filenames).to(device).long()
        outputs_cancer = net_cancer(inputs)
        outputs_cancer_atked = net_cancer(adv_images)


        # see if attack transfers to other model
        # net.load_state_dict(torch.load(checkpoint_paths_extra[labeling])['net'])
        # outputs2 = net(adv_images).detach()
        # net.load_state_dict(torch.load(checkpoint_path)['net'])
    
        loss_cancer = criterion(outputs_cancer, targets_cancer)            
        loss_cancer_atked = criterion(outputs_cancer_atked, targets_cancer)
        # Calculate predicted labels
        _, predicted_cancer = torch.max(outputs_cancer.data, 1)
        _, predicted_cancer_atked = torch.max(outputs_cancer_atked.data, 1)
        # Calculate accuracy
        total_examples_cancer += targets_cancer.size(0)
        correct_examples_cancer += (predicted_cancer == targets_cancer).sum().item()
        correct_examples_cancer_atked += (predicted_cancer_atked == targets_cancer).sum().item()

        targs_cancer += targets_cancer.cpu().tolist()
        preds_cancer += predicted_cancer.cpu().tolist()
        preds_cancer_atked += predicted_cancer_atked.cpu().tolist()

        if not vizzed:
            for j in range(num_viz):
                eg_atks[:, j, :, :, :] = torch.stack((inputs[j], adv_images[j], 10*(inputs[j] - adv_images[j])))

            vizzed = True

        # Calculate predicted labels (for GE vs Siemens classif.)
        _, predicted = outputs.max(1)
        total_examples += predicted.size(0)
        correct_examples += predicted.eq(targets).sum().item()
        test_loss += loss.item()

        # Calculate predicted labels (for GE vs Siemens classif. on other model)
        #_, predicted_othermodel = outputs2.max(1)
        #correct_examples_othermodel += predicted_othermodel.eq(targets).sum().item()



    # adv atk success on GE vs. Siemens model
    atk_acc = correct_examples / total_examples
    atk_auc = get_auc_score(targs, preds)
    print("adv acc = {}, adv auc = {}".format(atk_acc, atk_auc))

    # adv atk success on other model
    #atk_acc_othermodel = correct_examples_othermodel / total_examples
    #print("adv acc on other model = {}".format(atk_acc_othermodel))

    # effects of attack on cancer detection model
    cancer_acc = correct_examples_cancer / total_examples_cancer
    cancer_acc_atked = correct_examples_cancer_atked / total_examples_cancer

    print("cancer acc = {}, cancer acc atked = {}".format(cancer_acc, cancer_acc_atked))
    cancer_auc = get_auc_score(targs_cancer, preds_cancer)
    cancer_auc_atked = get_auc_score(targs_cancer, preds_cancer_atked)
    print("cancer auc = {}, cancer auc atked = {}".format(cancer_auc, cancer_auc_atked))

    # create example attack plot
    nrow = num_viz 
    eg_atks = eg_atks.reshape(nrow*3, *eg_atks.shape[2:])
    #save_image((eg_atks - eg_atks.min()) / (eg_atks.max() - eg_atks.min()), "results/imgs/eg_atks.png", nrow=nrow) 
    eg_atks_norm = (eg_atks - eg_atks.min()) / (eg_atks.max() - eg_atks.min())

    grid = make_grid(eg_atks_norm[:2*num_viz], nrow=nrow)
    plt.figure(figsize=(nrow*3, 2*3))
    plt.imshow(torch.permute(grid, (1, 2, 0)).cpu(), cmap="gray")
    plt.axis('off')
    plt.savefig(os.path.join(viz_folder, "eg_atks_onlabel{}_imgs.pdf".format(tgt_lbl)), bbox_inches = "tight")

    plt.figure(figsize=(nrow*3, 1*3))
    grid = make_grid(eg_atks_norm[:num_viz], nrow=nrow)
    grid = grid[0]
    plt.imshow(grid.cpu(), cmap='gray')
    grid = make_grid(eg_atks_norm[2*num_viz:], nrow=nrow)
    grid = grid[0]

    from matplotlib.colors import LogNorm
    im = plt.imshow(grid.cpu(), cmap='jet', alpha=0.8, norm=LogNorm(vmin=0.1, vmax=1))#, interpolation="gaussian")
    plt.colorbar(im)
    plt.axis('off')
    plt.savefig(os.path.join(viz_folder, "eg_atks_onlabel{}_atks.pdf".format(tgt_lbl)), bbox_inches = "tight")
    # plot accuracies
    # for dset_idx in range(len(dataset_names)):
    #     for i in range(2):
    #         plt.text(20 + dset_idx * (img_size + 2),
    #                 30 + i*(img_size + 2),
    #                 "acc. = {}%".format(round(100 * [eg_clean_accs, eg_atk_accs][i][dset_idx], 2)),
    #                 color="k",
    #                 weight="bold",
    #                 bbox=dict(facecolor=['cornflowerblue', 'lightcoral'][i], edgecolor='w')
    #                 )





# visualize errors
if full_test:
    if task == 'regression':  
        # errors
        plt.scatter(slice_locations, np.abs(all_preds-all_gts), s=10, alpha=0.75)
        plt.xlabel('slice location')
        plt.ylabel('error')
        plt.title('{} prediction Error vs. slice location\non test set (avg. error: {:.4f})'.format(labeling, avg_err))
        plt.show()

        # prediction vs. g.t.
        xline = np.linspace(np.min(all_gts), np.max(all_gts), 1000)
        plt.plot(xline, xline, 'k-')
        plt.scatter(all_gts, all_preds, s=10, alpha=0.75)
        plt.xlabel('true value')
        plt.ylabel('predicted value')
        plt.title('{} prediction vs. g.t. \non test set (avg. error: {:.4f})'.format(labeling, avg_err))
        plt.show()


# In[ ]:





