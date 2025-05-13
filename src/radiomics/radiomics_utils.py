#from pyradiomics.radiomics import featureextractor
from radiomics import featureextractor
#try:
#    from radiomics import featureextractor
#except ImportError:
#    import sys
#    sys.path.append('pyradiomics')
#    from radiomics import featureextractor
#from torchradiomics import (TorchRadiomicsFirstOrder,
#                            TorchRadiomicsGLCM,
#                            TorchRadiomicsGLDM,
#                            TorchRadiomicsGLRLM, TorchRadiomicsNGTDM,
#                            inject_torch_radiomics, restore_radiomics)
#
import SimpleITK as sitk
import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from multiprocess import Pool
from random import sample
import logging
import warnings
from sklearn.metrics import roc_auc_score

logger = logging.getLogger('radiomics.imageoperations')
logger.setLevel(logging.ERROR)


def compute_slice_radiomics(img_slice, mask_slice, params_file='/mnt/data1/breastHarmProj/domainshift_analysis/src/radiomics/configs/2D_extraction.yaml'):
    device = 'cpu' # GPU implementation is too slow, possibly bugged
    # device = 'cuda'

    # assume pixel size is in 1x1 mm. I am gonna assume 3rd dimension which is the depth is also 1 mm so that it is isotropic
    data_spacing = [1,1,1]

    sitk_img = sitk.GetImageFromArray(img_slice)
    sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
    sitk_img = sitk.JoinSeries(sitk_img)

    sitk_mask = sitk.GetImageFromArray(mask_slice)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2]) ))
    sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32) # MAKE SURE IT IS CASTED IN INT

    if str(device) == 'cuda':
        print("Using GPU for radiomics")
        inject_torch_radiomics() # replace cpu version with torch version

        # prepare the settings and load
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file, device=device)

        #extract 
        features = extractor.execute(sitk_img, sitk_mask)

        restore_radiomics() # restore
    else:
        # prepare the settings and load
        extractor = featureextractor.RadiomicsFeatureExtractor(params_file)

        #extract 
        features = extractor.execute(sitk_img, sitk_mask)

    return features

def get_IAPs(img_fname, IAP_names):
    clinical_features_path = '../data/dbc/maps/Clinical_and_Other_Features.csv'
    clinical_features = pd.read_csv(clinical_features_path)
    img_fname = img_fname.replace("_", "-")
    img_patient_id = "Breast_MRI_{}".format(img_fname.split('-')[2].replace(".png", ""))

    IAPs = {}
    for IAP_name in IAP_names:
        IAPs[IAP_name] = clinical_features[clinical_features['Patient ID'] == img_patient_id][IAP_name].item()
    return img_patient_id, IAPs

    #feats1 = radiomics_df.to_numpy().astype(np.float32)
    #feats2 = real_radiomics_df.to_numpy().astype(np.float32)

def convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                    radiomics_df2,
                                    match_sample_count=False,
                                    return_image_fnames=False,
                                    return_feature_names=False,
                                    normalization='zscore_bysourcedomain',
                                    exclude_features=None
                                    ):
    """
    Convert radiomics dataframes to numpy arrays and normalize them wrt the real radiomics data
    also possibly remove features that are NaN in either of the arrays, and use random removal to match the sample count
    """
    imgfnames1 = radiomics_df1['img_fname'].values
    imgfnames2 = radiomics_df2['img_fname'].values

    # exclude shape-based radiomics which are for object-level radiomics
    # iterate through column names
    for col in radiomics_df1.columns:
        check = "shape2D" in col
        if check:
            radiomics_df1 = radiomics_df1.drop(columns=col)
            radiomics_df2 = radiomics_df2.drop(columns=col)

    if exclude_features is not None:
        if exclude_features == "textural":
            print("EXCLUDING TEXTURAL RADIOMICS.")
            # iterate through column names
            for col in radiomics_df1.columns:
                colsplit = col.split('_')
                check = (colsplit[0].split('-')[0] in ["original", "wavelet"]) and (colsplit[1].startswith("gl"))
                if check:
                    radiomics_df1 = radiomics_df1.drop(columns=col)
                    radiomics_df2 = radiomics_df2.drop(columns=col)
        
        elif exclude_features.startswith("wavelet"):
            if exclude_features == "wavelet":
                print("EXCLUDING WAVELET RADIOMICS.")
                # iterate through column names
                for col in radiomics_df1.columns:
                    check = col.startswith("wavelet")
                    if check:
                        radiomics_df1 = radiomics_df1.drop(columns=col)
                        radiomics_df2 = radiomics_df2.drop(columns=col)
            else:
                print("EXCLUDING {} RADIOMICS.".format(exclude_features))
                # iterate through column names
                for col in radiomics_df1.columns:
                    check = col.startswith(exclude_features)
                    if check:
                        radiomics_df1 = radiomics_df1.drop(columns=col)
                        radiomics_df2 = radiomics_df2.drop(columns=col)

        elif exclude_features == "firstorder":
            print("EXCLUDING FIRST ORDER RADIOMICS.")
            # iterate through column names
            for col in radiomics_df1.columns:
                check = "firstorder" in col
                if check:
                    radiomics_df1 = radiomics_df1.drop(columns=col)
                    radiomics_df2 = radiomics_df2.drop(columns=col)

        else:
            raise NotImplementedError("Invalid exclude_features argument")
        
    # remove NaN and string radiomics
    # Identify columns with string data type
    radiomics_df1 = radiomics_df1.drop(columns=radiomics_df1.select_dtypes(include=['object']).columns)
    radiomics_df2 = radiomics_df2.drop(columns=radiomics_df2.select_dtypes(include=['object']).columns)

    radiomics_df1 = radiomics_df1.dropna()
    radiomics_df2 = radiomics_df2.dropna()

    if return_feature_names:
        assert radiomics_df1.columns.equals(radiomics_df2.columns)
        feature_names = radiomics_df1.columns

    # convert radiomics to arrays
    feats1 = radiomics_df1.to_numpy().astype(np.float32)
    feats2 = radiomics_df2.to_numpy().astype(np.float32)

    # match first dimension by random removal
    if match_sample_count:
        if feats1.shape[0] > feats2.shape[0]:
            mask = np.random.choice(feats1.shape[0], feats2.shape[0], replace=False)
            feats1 = feats1[mask]
            imgfnames1 = imgfnames1[mask]
        elif feats2.shape[0] > feats1.shape[0]:
            mask = np.random.choice(feats2.shape[0], feats1.shape[0], replace=False)
            feats2 = feats2[mask]
            imgfnames2 = imgfnames2[mask]

    print(feats1.shape, feats2.shape)
    # normalize features in these arrays wrt first feature dist
    if normalization == 'zscore_bysourcedomain':
        mean = np.mean(feats1, axis=0)
        std = np.std(feats1, axis=0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            feats1 = (feats1 - mean) / std
            feats2 = (feats2 - mean) / std

        #print("USING IMAGENET STATISTICS, FOR TESTING ONLY!")
        #FIDmu = 0.38
        #FIDstd = 0.28
        #feats1 = FIDstd*feats1 + FIDmu
        #feats2 = FIDstd*feats2 + FIDmu


    elif normalization == "zscore_separate":
        feats1 = zscore_normalization(feats1)
        feats2 = zscore_normalization(feats2)

    elif normalization == 'frd':
        # as implemented for FRD https://arxiv.org/abs/2403.13890 
        feats1 = frd_normalization(feats1)
        feats2 = frd_normalization(feats2)

    else:
        raise NotImplementedErro

    # remove features from both arrays that are NaN or inf in either
    nan_features_mask = np.isnan(feats1).any(axis=0) | np.isnan(feats2).any(axis=0) | np.isinf(feats1).any(axis=0) | np.isinf(feats2).any(axis=0)
    feats1 = feats1[:, ~nan_features_mask]
    feats2 = feats2[:, ~nan_features_mask]
    if return_feature_names:
        feature_names = feature_names[~nan_features_mask]
        assert len(feature_names) == feats1.shape[1] and len(feature_names) == feats2.shape[1]

    ret = [feats1, feats2]
    if return_image_fnames:
        assert len(imgfnames1) == feats1.shape[0] and len(imgfnames2) == feats2.shape[0]
        ret += [imgfnames1, imgfnames2]
    
    if return_feature_names:
        ret += [feature_names]
    
    return ret


def compute_and_save_imagefolder_radiomics(
        img_dir,
        radiomics_fname='radiomics.csv',
        subset=None
):
    out_dir = img_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    radiomics_csv_fname = os.path.join(out_dir, radiomics_fname)

    img_filenames = []
    radiomics = []

    img_fnames = os.listdir(img_dir)
    if subset is not None:
        img_fnames = sample(img_fnames, subset)

    # pyradiomics usage with numpy following https://github.com/AIM-Harvard/pyradiomics/issues/449
    # assume 2D images here
    for img_idx, img_fname in tqdm(enumerate(img_fnames), total=len(img_fnames)):
        #if img_idx > 10:
        #    break

        if not img_fname.split('.')[-1] in ['png', 'jpeg', 'jpg']:
            continue
        img_slice = np.asarray(Image.open(os.path.join(img_dir, img_fname)).convert('L')).copy()
        mask_slice = np.ones_like(img_slice)
        # to prevent bug in pyradiomics https://github.com/AIM-Harvard/pyradiomics/issues/765#issuecomment-1116713745
        mask_slice[0][0] = 0

        features = {}
        try:
            features = compute_slice_radiomics(img_slice, mask_slice)
        except RuntimeError:
            continue

        radiomics.append(features)
        img_filenames.append(img_fname)
    
    # bring all data together and save
    radiomics_df = pd.DataFrame(radiomics)
    radiomics_df.insert(loc=0, column='img_fname', value=img_filenames)
    # save df as a csv
    radiomics_df.to_csv(radiomics_csv_fname, index=False)

    print("saved radiomics to {}".format(radiomics_csv_fname))

    return radiomics_df

def compute_and_save_imagefolder_radiomics_parallel(
        img_dir,
        radiomics_fname='radiomics.csv',
        subset=None,
        num_workers=8
):
    out_dir = img_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    radiomics_csv_fname = os.path.join(out_dir, radiomics_fname)
    
    img_fnames = os.listdir(img_dir)
    if subset is not None:
        img_fnames = sample(img_fnames, subset)

    def get_radiomics_feature(split, img_list):
        radiomics = []
        img_filenames = []
        split_num = len(img_list) // num_workers
        img_list = img_list[split_num * split : split_num * (split + 1)]

        # pyradiomics usage with numpy following https://github.com/AIM-Harvard/pyradiomics/issues/449
        # assume 2D images here
        for img_idx, img_fname in enumerate(tqdm(img_list, total=len(img_list))):
            #if img_idx > 5:
            #    break
            if not img_fname.split('.')[-1] in ['png', 'jpeg', 'jpg']:
                continue
            img_slice = np.asarray(Image.open(os.path.join(img_dir, img_fname)).convert('L')).copy()
            mask_slice = np.ones_like(img_slice)
            # to prevent bug in pyradiomics https://github.com/AIM-Harvard/pyradiomics/issues/765#issuecomment-1116713745
            mask_slice[0][0] = 0

            features = {}
            try:
                features = compute_slice_radiomics(img_slice, mask_slice)
            except RuntimeError:
                continue
            radiomics.append(features)
            img_filenames.append(img_fname)

        return radiomics, img_filenames

    # Multi-Process
    pool = Pool()
    result_list = []
    imgs = os.listdir(img_dir)
    for i in range(num_workers):
        result = pool.apply_async(get_radiomics_feature, [i, img_fnames])
        result_list.append(result)
    
    radiomics = []
    img_filenames = []
    num_skipped = 0
    for r in result_list:
        try:
            radiomics_sub, filenames_sub = r.get(timeout=100000)
            radiomics += radiomics_sub
            img_filenames += filenames_sub
        except ValueError:
            num_skipped += 1
    
    # bring all data together and save
    radiomics_df = pd.DataFrame(radiomics)
    radiomics_df.insert(loc=0, column='img_fname', value=img_filenames)
    # save df as a csv
    radiomics_df.to_csv(radiomics_csv_fname, index=False)

    print("saved radiomics to {}".format(radiomics_csv_fname))
    if num_skipped != 0:
        print("had to skip {} images due to errors.".format(num_skipped))

    return radiomics_df

# dataloader for radiomic features

def compute_normalized_RaD(feats1, feats2, val_frac=0.1):

    # randomly split feats1 into train (reference set) and val (Establish distance dist), via val_frac
    val_idx = np.random.choice(feats1.shape[0], int(val_frac*feats1.shape[0]), replace=False)
    train_idx = np.array([i for i in range(feats1.shape[0]) if i not in val_idx])
    in_activations_val = feats1[val_idx]
    in_activations = feats1[train_idx]

    in_activations = torch.tensor(in_activations)
    in_activations_val = torch.tensor(in_activations_val)
    feats2 = torch.tensor(feats2)
    
    id_mean = in_activations.mean(dim=0)

    ID_scores_val = torch.stack([torch.norm(id_mean - out, dim=0) for out in in_activations_val])
    ID_scores_val = ID_scores_val.detach().numpy()

    scores = torch.stack([torch.norm(id_mean - out, dim=0) for out in feats2])
    scores = scores.detach().numpy()

    all_scores = np.concatenate([ID_scores_val, scores])
    all_labels = np.concatenate([np.zeros(len(ID_scores_val)), np.ones(len(scores))])
    auc_dev = 2. * np.abs(roc_auc_score(all_labels, all_scores) - 0.5)

    return auc_dev

# misc
def frd_normalization(feats):
    # minmax normalize
    feats = (feats - np.min(feats, axis=0)) / (np.max(feats, axis=0) - np.min(feats, axis=0))
    # scale by typical FID feature range
    FID_featsmax = 7.45670747756958
    feats = feats * FID_featsmax
    return feats

def zscore_normalization(feats):
    mean = np.mean(feats, axis=0)
    std = np.std(feats, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        feats = (feats - mean) / std

    return feats
