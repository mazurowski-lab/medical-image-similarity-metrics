"""
Compute and interpret radiomic distances between two datasets.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from argparse import ArgumentParser
from time import time

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skdim import id
from geomloss import SamplesLoss

from src.radiomics.radiomics_utils import convert_radiomic_dfs_to_vectors, compute_and_save_imagefolder_radiomics, compute_and_save_imagefolder_radiomics_parallel, compute_normalized_RaD
from src.utils import *

viz_folder = "visualizations"

# use serif font
#plt.rcParams['font.family'] = 'serif'

def plot_tsne(feats, labels, feature_name, domain_name="Manufacturer", domain_type="categorical"):
    print("Running low dim embedding on features...")

    for emb_type in ["TSNE", "PCA"]:
        if emb_type == "TSNE":
            emb = TSNE(
                n_components=2, perplexity=10, n_iter=10000, verbose=True
                ).fit_transform(
                    feats
                                )
        else:
            emb = PCA(n_components=2).fit_transform(feats)

        # plot with seaborn with domain labels
        plt.figure(figsize=(6,6), dpi=300)
        if domain_type == "categorical":
            g = sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette="tab10", alpha=0.5)
        elif domain_type == "continuous":
            vmin, vmax = np.min(labels), np.max(labels)
            cmap = plt.cm.viridis

            norm=plt.Normalize(vmin=vmin, vmax=vmax)

            g = sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=labels, palette=cmap) 
        else:
            raise ValueError("domain_type must be 'categorical' or 'continuous'")
        plt.title("{} feature representations\n domain: {}".format(feature_name, domain_name))

        if domain_type == "categorical":
            legend_handles, _= g.get_legend_handles_labels()
            g.legend(legend_handles, ['1', '2'], 
                bbox_to_anchor=(1,1), 
                title=domain_name)
        
        plt.savefig(os.path.join(viz_folder, "{}_emb_{}_{}.png".format(feature_name, emb_type, domain_name)), bbox_inches="tight")

def interpret_radiomic_differences(
        radiomics_path1,
        radiomics_path2,
        run_tsne = True,
):

    # load radiomics and convert to numpy arrays
    radiomics_df1 = pd.read_csv(radiomics_path1)
    radiomics_df2 = pd.read_csv(radiomics_path2)
    feats1, feats2, imgfnames1, imgfnames2, feature_names = convert_radiomic_dfs_to_vectors(radiomics_df1, radiomics_df2, return_image_fnames=True, return_feature_names=True)
    # note: feats are normalized wrt the first radiomics df

    domain_type = "categorical"
    do_special_domain_plotting = False
    if "dbc/prior_work_1k" in radiomics_path1 and do_special_domain_plotting:
        # custom domain labels
        domain_name = "FOV Computed (Field of View) in cm "
        domain_type = "continuous"

        clinical_feats_df = pd.read_csv('data/dbc/maps/Clinical_and_Other_Features.csv')
        imgfnames = np.concatenate([imgfnames1, imgfnames2])
        patient_ids = ['Breast_MRI_{}'.format(filename.split('-')[2].replace('.png', '')) for filename in imgfnames]
        tsne_labels = [clinical_feats_df[clinical_feats_df['Patient ID'] == patient_id][domain_name].item() for patient_id in patient_ids]
        tsne_labels = np.array(tsne_labels)
    
    else: 
        domain_name = "Manufacturer"
        domain_labels1 = np.zeros(feats1.shape[0])
        domain_labels2 = np.ones(feats2.shape[0])
        tsne_labels = np.concatenate([domain_labels1, domain_labels2])

    # visualize radiomic feature representations using t-SNE 
    if run_tsne:
        also_plot_FID_features = False
        all_feats = {'radiomic': (feats1, feats2)}
        if also_plot_FID_features:
            # do the same for saved FID/RadFID features
            fid_feats1 = np.load(radiomics_path1.replace("radiomics.csv", "FID_acts.npy"))
            fid_feats2 = np.load(radiomics_path2.replace("radiomics.csv", "FID_acts.npy"))
            all_feats['FID'] = (fid_feats1, fid_feats2)

            radfid_feats1 = np.load(radiomics_path1.replace("radiomics.csv", "RadFID_acts.npy"))
            radfid_feats2 = np.load(radiomics_path2.replace("radiomics.csv", "RadFID_acts.npy"))
            all_feats['RadFID'] = (radfid_feats1, radfid_feats2)
        
        for feature_name, (f1, f2) in all_feats.items():
            feats = np.concatenate([f1, f2])
            plot_tsne(feats, tsne_labels, feature_name, domain_name=domain_name, domain_type=domain_type)


    # in case some images are jpegs and others are pngs,
    imgfnames1 = np.array([os.path.splitext(fname)[0] for fname in imgfnames1])
    imgfnames2 = np.array([os.path.splitext(fname)[0] for fname in imgfnames2])
    radiomics_df1['img_fname'] = [os.path.splitext(fname)[0] for fname in radiomics_df1['img_fname']]

    # if images are paired (i.e. between source domain and translated domain):
    # only use features that are present in both dataframes
    final_feats1 = []
    final_feats2 = []
    final_imgfnames = []
    for img_fname in radiomics_df1['img_fname'].values:
        # print(img_fname in imgfnames1, img_fname in imgfnames2)
        if img_fname in imgfnames1 and img_fname in imgfnames2:
            final_feats1.append(feats1[imgfnames1 == img_fname])
            final_feats2.append(feats2[imgfnames2 == img_fname])
            final_imgfnames.append(img_fname)

    if len(final_imgfnames) > 0:
        print("Dataset is paired, analyzing...".format(len(final_imgfnames)))

        feats1 = np.array(final_feats1).squeeze()
        feats2 = np.array(final_feats2).squeeze()
        imgfnames = np.array(final_imgfnames)

        # find images that differ most in radiomics between the two features
        # in squared l2 norm
        difference = feats1 - feats2

        # plot distribution of squared l2 radiomic differences across all images, 
        # averaged across all radiomic features
        avg_abs_differences = np.mean(difference**2, axis=1)
        sorted_indices = avg_abs_differences.argsort()[::-1]
        x = np.arange(len(avg_abs_differences))
        y = avg_abs_differences[sorted_indices]
        plt.figure(figsize=(4,2), dpi=300)
        plt.plot(x, y, '-', color='cornflowerblue')
        # plt.xscale('log')
        #plt.yscale('log')

        plt.title('Which images changed\nthe most/least through translation?')
        plt.xlabel('sorted image index')
        plt.ylabel('$||\Delta h||_2$ (img. level)')
        plt.grid()
        plt.savefig(os.path.join(viz_folder, "sorted_radiomic_image_differences.png"), bbox_inches="tight")
        plt.show()


        # plot the images that differ most and least
        num_images = 4
        fig, axs = plt.subplots(num_images//2, 2, figsize=(3, num_images*0.75), dpi=300)

        axs[0, 0].text(0, -50, "Input->Output images that changed\nthe most through translation:", fontsize=8)
        # plot images with highest differences
        show_image_fnames = False
        for i in range(num_images//2):
            idx = sorted_indices[i]

            img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img1_fname):
                img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"
            
            img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img2_fname):
                img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"
            
            img1 = plt.imread(img1_fname)
            img2 = plt.imread(img2_fname)

            axs[i, 0].imshow(img1, cmap='gray')
            axs[i, 1].imshow(img2, cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            if show_image_fnames:
                axs[i, 0].set_title(imgfnames[idx])

        plt.savefig(os.path.join(viz_folder, "images_with_highest_radiomic_differences.png"), bbox_inches="tight")
        plt.close()
                
        fig, axs = plt.subplots(num_images//2, 2, figsize=(3, num_images*0.75), dpi=300)
        axs[0, 0].text(0, -50, "Images that changed\nthe least:", fontsize=8)
        for i in range(num_images//2):
            idx = sorted_indices[len(sorted_indices) - 1 - (num_images//2) + i]
            #print(len(sorted_indices) - 1 - (num_images//2) + i)
            img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img1_fname):
                img1_fname = os.path.join(radiomics_path1.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"
            
            img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".png"
            if not os.path.exists(img2_fname):
                img2_fname = os.path.join(radiomics_path2.replace("/radiomics.csv",""), imgfnames.tolist()[idx]) + ".jpg"

            img1 = plt.imread(img1_fname)
            img2 = plt.imread(img2_fname)

            axs[i, 0].imshow(img1, cmap='gray')
            axs[i, 1].imshow(img2, cmap='gray')
            axs[i, 0].axis('off')
            axs[i, 1].axis('off')
            if show_image_fnames:
                axs[i, 0].set_title(imgfnames[idx])

        plt.savefig(os.path.join(viz_folder, "images_with_lowest_radiomic_differences.png"), bbox_inches="tight")

    # do the same type of plot but for each radiomic feature, averaged across all images
    # (see which radiomic features differ most between the two datasets in terms of squared l2 norm)

    take_full_frechet_distance = True
    if take_full_frechet_distance:
        # compute distance with frechet distance for each feature, over all images
        avg_abs_differences = []
        for feat_idx in tqdm(range(feats1.shape[1])):
            avg_abs_differences.append(
                frechet_distance(feats1[:, feat_idx], feats2[:, feat_idx])
            )
        avg_abs_differences = np.array(avg_abs_differences)
    else:
        avg_abs_differences = np.mean(difference**2, axis=0)
    sorted_indices = avg_abs_differences.argsort()[::-1]
    x = np.arange(len(avg_abs_differences))
    y = avg_abs_differences[sorted_indices]
    plt.figure(figsize=(4,2), dpi=300)
    plt.plot(x, y, '-', color='indianred')
    # plt.xscale('log')
    plt.yscale('log')

    plt.title('How many radiomic features\nchanged noticeably through translation?')
    plt.xlabel('sorted feature index $j$')
    plt.ylabel('$|\Delta h|^j$ (dist. level)')
    plt.grid()
    # plot vertical line capturing 90% of the total difference
    total_diff = np.sum(y)
    ninety_percent_diff = 0.5 * total_diff
    cumsum_diff = np.cumsum(y)
    ninety_percent_idx = np.argmax(cumsum_diff > ninety_percent_diff)
    print("X% of the total difference is captured by the first {} features.".format(ninety_percent_idx))
    plt.axvline(x=ninety_percent_idx, color='k', linestyle='--')

    plt.savefig(os.path.join(viz_folder, "sorted_radiomic_feature_differences.png"), bbox_inches="tight")

    # get feature names in order of sorted indices
    sorted_feature_names = feature_names[sorted_indices]
    k = 10
    # print top k changed features
    print("Top {} changed radiomic features:".format(k))

    for i in range(k):
        print('{}'.format(sorted_feature_names[i]))

    print("with log differences:")
    for i in range(k):
        print('{}: {:.1f}'.format(sorted_feature_names[i], np.log(y[i])))

    return


def main(
        image_folder1,
        image_folder2,
        interpret = False,
        compute_auc_deviation = True,
        run_tsne = False,
        feature_weights_path = None,
        subset = None,
        compute_fresh = False,
        exclude_features = False,
        normalization='zscore_bysourcedomain',
        compute_onlyforsingledist = None,
        device = 'cuda',
        parallelize = True
):
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        torch.multiprocessing.set_start_method('spawn', force=True)

    radiomics_fname = 'radiomics.csv'
    if subset is not None:
        radiomics_fname = radiomics_fname.replace(".csv", "_subset{}.csv".format(subset))

    radiomics_path1 = os.path.join(image_folder1, radiomics_fname)
    radiomics_path2 = os.path.join(image_folder2, radiomics_fname)

    # if needed, compute radiomics for the images
    if compute_fresh or not os.path.exists(radiomics_path1):
        print("No radiomics found computed for image folder 1 at {}, computing now.".format(radiomics_path1))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder1, radiomics_fname=radiomics_fname, subset=subset)
        else:
            compute_and_save_imagefolder_radiomics(image_folder1, radiomics_fname=radiomics_fname, subset=subset)
        print("Computed radiomics for image folder 1.")
    else:
        print("Radiomics already computed for image folder 1 at {}.".format(radiomics_path1))

    if compute_fresh or not os.path.exists(radiomics_path2):
        print("No radiomics found computed for image folder 2 at {}, computing now.".format(radiomics_path2))
        if parallelize:
            compute_and_save_imagefolder_radiomics_parallel(image_folder2, radiomics_fname=radiomics_fname, subset=subset)
        else:
            compute_and_save_imagefolder_radiomics(image_folder2, radiomics_fname=radiomics_fname, subset=subset)
        print("Computed radiomics for image folder 2.")
    else:
        print("Radiomics already computed for image folder 2 at {}.".format(radiomics_path2))

    # load radiomics
    radiomics_df1 = pd.read_csv(radiomics_path1)
    radiomics_df2 = pd.read_csv(radiomics_path2)

    feature_exclusions = [None]
    feature_exclusion_RaDs = []
    if exclude_features:
        feature_exclusions = ["textural", "wavelet", "firstorder"]
        #feature_exclusions = [["wavelet-LL", "wavelet-LH", "wavelet-HL", "wavelet-HH"][3]]

    for feature_exclusion in feature_exclusions:
        feats1, feats2 = convert_radiomic_dfs_to_vectors(radiomics_df1, 
                                                             radiomics_df2,
                                                             match_sample_count=True, # needed for distance measures
                                                             exclude_features=feature_exclusion,
                                                             normalization=normalization
                                                             ) 

        if feature_weights_path is not None:
            print("NOTE: Using feature weights from {} to compute radiomic distances.".format(feature_weights_path))
            feature_weights = torch.load(feature_weights_path)['w'].cpu().numpy()

            hard_threshold = 0.5
            if hard_threshold is not None:
                print("completely excluding features with weights below threshold.")
                feature_weights = (feature_weights > hard_threshold).astype(float)
                feats1 = feats1[:, feature_weights > 0]
                feats2 = feats2[:, feature_weights > 0]
            else:
                feats1 = feats1 * feature_weights
                feats2 = feats2 * feature_weights


        #if subset is not None:
        #    assert subset <= feats1.shape[0], "subset size must be less than or equal to number of images in dataset 1, but got {} > {}".format(subset, feats1.shape[0])
        #    print("Using a random subset of {} images for analysis.".format(subset))
        #    subset_indices = np.random.choice(feats1.shape[0], subset, replace=False)
        #    feats1 = feats1[subset_indices]
        #    feats2 = feats2[subset_indices]


        # print number of rows which have any NaN or Inf values
        # print("Number of rows with NaN or Inf values in radiomics:")
        # print("Dataset 1: {}".format(np.sum(np.isnan(feats1) | np.isinf(feats1))))
        # print("Dataset 2: {}".format(np.sum(np.isnan(feats2) | np.isinf(feats2))))

        # # number of columns with any NaN or Inf values
        # print("Number of columns with NaN or Inf values in radiomics:")
        #print("Dataset 1: {}".format(np.sum(np.isnan(feats1) | np.isinf(feats1), axis=0)))
        #print("Dataset 2: {}".format(np.sum(np.isnan(feats2) | np.isinf(feats2), axis=0)))


        # compute radiomic distances
        print("radiomic distance measures:")

        if compute_auc_deviation:
            # average auc deviation
            auc_devs = []
            for i in range(feats1.shape[1]):
                all_feat_values = np.concatenate([feats1[:, i], feats2[:, i]])
                labels = np.concatenate([np.zeros(feats1.shape[0]), np.ones(feats2.shape[0])])
                auc = roc_auc_score(labels, all_feat_values)
                auc_devs.append(np.abs(auc - 0.5))

            print("avg and median AUC deviation: \n{}\n{}".format(np.mean(auc_devs), np.median(auc_devs)))

        if compute_onlyforsingledist is not None:
            # special for testing only: compute statistic of just dist 1
            stat_type = compute_onlyforsingledist
            stat = None
            if stat_type == "intrinsic_dim":
                # intrinsic dimension
                intrinsic_dim_estimator = id.MLE
                #intrinsic_dim_estimator = id.TwoNN
                #intrinsic_dim_estimator = id.lPCA
                #intrinsic_dim_estimator = id.KNN

                stat = intrinsic_dim_estimator().fit(X=feats1).transform()
            elif stat_type == "logdetcov":
                #pcadim = np.linalg.matrix_rank(np.cov(feats1, rowvar=False))
                #feats1 = PCA(n_components=pcadim).fit_transform(feats1)
                cov = np.cov(feats1, rowvar=False)

                # compute log determinant of covariance matrix
                stat = np.log(np.linalg.det(cov))
            else:
                raise ValueError("stat_type must be 'intrinsic_dim' or 'logdetcov'")

            print("domain1 {}: {}".format(stat_type, stat))

            return stat

        # Compute the Wasserstein distance between the two datasets
        # Create a SamplesLoss object with the 'sinkhorn' algorithm for efficient computation
        loss = SamplesLoss("sinkhorn")

        # convert np arrays to torch tensors
        feats1 = torch.tensor(feats1).to(device)
        feats2 = torch.tensor(feats2).to(device)

        try:
            wasserstein_distance = loss(feats1, feats2)
            print("Wasserstein distance:", wasserstein_distance.item())
        except ValueError as e:
            print(e)


        feats1 = feats1.cpu().numpy()
        feats2 = feats2.cpu().numpy()

        print(feats1.shape, feats2.shape)
        # Frechet distance
        fd = frechet_distance(feats1, feats2)

        feature_exclusion_RaDs.append(fd)

        compute_relative_rad = False
        if compute_relative_rad:
            # compute relative RaD:
            # randomly split feats1 into two halves, compute the Frechet distance between the two halves
            # and divide the original Frechet distance by this value
            # do same, but normalizing factor is averaged over taking different halfs of the dataset
            num_splits = 10
            fd_halfs1 = []
            for i in range(num_splits):
                # do random split
                feats1_shuffled = np.random.permutation(feats1)
                feats1_half1, feats1_half2 = np.array_split(feats1_shuffled, 2, axis=0) 
                fd_half1 = frechet_distance(feats1_half1, feats1_half2)
                fd_halfs1.append(fd_half1)

            print("{} +/- {}".format(np.mean(np.log(fd_halfs1)), np.std(np.log(fd_halfs1))))
            relative_RaD = np.log(fd) / np.mean(np.log(fd_halfs1))
            print("Relative RaD with log (averaged over {} splits): {}".format(num_splits, relative_RaD)) 

        compute_rad_tests = False
        if compute_rad_tests:
            fd_meanonly = frechet_distance(feats1, feats2, means_only=True)
            rad_meanonly = np.log(fd_meanonly)
            print("RaD (mean terms only, log included): {}".format(rad_meanonly))

        compute_nRaD = False
        if compute_nRaD:
            nRaD = compute_normalized_RaD(feats1, feats2)
            print("Normalized RaD: {}".format(nRaD))


        # MMD
        # convert np arrays to torch tensors
        feats1 = torch.tensor(feats1).to(device)
        feats2 = torch.tensor(feats2).to(device)
        mmd = np.sqrt(MMDLoss()(feats1, feats2).item())
        print("MMD: {}".format(mmd))

        # interpret and evaluate radiomic differences
        if interpret:
            interpret_radiomic_differences(radiomics_path1, radiomics_path2, run_tsne=run_tsne)


    if len(feature_exclusions) > 1:
        ret = []

    if normalization == 'frd':
        print("RaD results, {} normalization (no log):".format(normalization))
        for r in feature_exclusion_RaDs:
            print(r)
            if len(feature_exclusions) > 1:
                ret.append(r)
            else:
                ret = r
    else:
        print("RaD results (with logarithm), {} normalization:".format(normalization))
        for r in feature_exclusion_RaDs:
            print(np.log(r))
            if len(feature_exclusions) > 1:
                ret.append(np.log(r))
            else:
                ret = np.log(r)

    return ret


if __name__ == "__main__":
    tstart = time()
    parser = ArgumentParser()

    parser.add_argument('--image_folder1', type=str, required=True)
    parser.add_argument('--image_folder2', type=str, required=True)
    parser.add_argument('--feature_weights_path', type=str, default=None, help='path to feature weights for distance model')
    parser.add_argument('--run_tsne', action='store_true')
    parser.add_argument('--subset', type=int, default=None, help='subset of images to use for analysis')
    parser.add_argument('--exclude_features', action='store_true', help='do feature exclusion experiments')
    parser.add_argument('--interpret', action='store_true', help='do interpretability study')
    parser.add_argument('--fresh', action='store_true', help='re-compute all radiomics fresh')
    parser.add_argument('--normalization', type=str, default="zscore_bysourcedomain")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = torch.cuda.is_available()
    print('running on {}'.format(device))

    main(
        args.image_folder1,
        args.image_folder2,
        interpret=args.interpret,
        run_tsne=args.run_tsne,
        feature_weights_path=args.feature_weights_path,
        exclude_features=args.exclude_features,
        subset=args.subset,
        compute_fresh=args.fresh,
        normalization=args.normalization,
        device=device
        )
        # image_folder1 = 'data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/Siemens',
        # image_folder2 = 'data/dbc/prior_work_1k/harmonized/by_unsb512/SiemenstoGE/fake_1',

    tend = time()
    print("compute time (sec) for N: {} {}".format(tend - tstart, args.subset))
