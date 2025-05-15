# Medical Image Similarity Metrics

#### By [Nicholas Konz*](https://nickk124.github.io/), [Richard Osuala*](https://scholar.google.com/citations?user=0KkVRVQAAAAJ&hl=en), (* = equal contribution), [Preeti Verma](https://scholar.google.com/citations?user=6WN41lwAAAAJ&hl=en), [Yuwen Chen](https://scholar.google.com/citations?user=61s49p0AAAAJ&hl=en), [Hanxue Gu](https://scholar.google.com/citations?user=aGjCpQUAAAAJ&hl=en), [Haoyu Dong](https://haoyudong-97.github.io/), [Yaqian Chen](https://scholar.google.com/citations?user=iegKFuQAAAAJ&hl=en), [Andrew Marshall](linkedin.com/in/andrewmarshall26), [Lidia Garrucho](https://github.com/LidiaGarrucho), [Kaisar Kushibar](https://scholar.google.es/citations?user=VeHqMi4AAAAJ&hl=en), [Daniel M. Lang](https://scholar.google.com/citations?user=AV04Hs4AAAAJ&hl=en), [Gene S. Kim](https://vivo.weill.cornell.edu/display/cwid-sgk4001), [Lars J. Grimm](https://scholars.duke.edu/person/lars.grimm), [John M. Lewin](https://medicine.yale.edu/profile/john-lewin/), [James S. Duncan](https://medicine.yale.edu/profile/james-duncan/), [Julia A. Schnabel](https://compai-lab.github.io/), [Oliver Diaz](https://sites.google.com/site/odiazmontesdeoca/home), and [Karim Lekadir](https://www.bcn-aim.org/)

arXiv paper link: [![arXiv Paper](https://img.shields.io/badge/arXiv-2412.01496-orange.svg?style=flat)](https://arxiv.org/abs/2412.01496)

<p align="center">
  <img src='https://github.com/mazurowski-lab/medical-image-similarity-metrics/blob/master/figs/evalframework.png' width='95%'>
</p>

We provide an easy-to-use framework for evaluating distance/similarity metrics between unpaired sets of medical images with a variety of metrics, accompanying our [paper](https://arxiv.org/abs/2412.01496). For example, this can be used to evaluate the performance of image generative models in the medical imaging domain. The codebase includes implementations of several distance metrics that can be used to compare images, as well as tools for evaluating the performance of generative models on various downstream tasks.

Included metrics:
1. [FRD](https://arxiv.org/abs/2412.01496) (Fréchet Radiomic Distance)
2. [FID](https://papers.nips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html) (Fréchet Inception Distance)
3. [Radiology FID/RadFID](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-10/issue-06/061403/medigan--a-Python-library-of-pretrained-generative-models-for/10.1117/1.JMI.10.6.061403.full)
4. [KID](https://openreview.net/forum?id=r1lUOzWCW) (Kernel Inception Distance)
5. [CMMD](https://arxiv.org/abs/2401.09603) (CLIP Maximum Mean Discrepancy)

## Credits

Thanks to the following repositories which this framework utilizes and builds upon:

1. [frd-score](https://github.com/RichardObi/frd-score)
2. [pyradiomics](https://github.com/AIM-Harvard/pyradiomics)
3. [gan-metrics-pytorch](https://github.com/abdulfatir/gan-metrics-pytorch), which we modified to allow for computing RadFID.
4. [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch)

## Citation

Please cite our paper if you use this framework in your work:

```bib
@article{konzosuala_frd2025,
      title={Fr\'echet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets}, 
      author = {Konz, Nicholas and Osuala, Richard and Verma, Preeti and Chen, Yuwen and Gu, Hanxue and Dong, Haoyu and Chen, Yaqian and Marshall, Andrew and Garrucho, Lidia and Kushibar, Kaisar and Lang, Daniel M. and Kim, Gene S. and Grimm, Lars J. and Lewin, John M. and Duncan, James S. and Schnabel, Julia A. and Diaz, Oliver and Lekadir, Karim and Mazurowski, Maciej A.},
      year={2025},
      eprint={2412.01496},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.01496}, 
}
```

## 0. Installation/Setup

1. First, note that Python <=3.9 is required due to one of the distances, FRD, using PyRadiomics (see [here](https://github.com/AIM-Harvard/pyradiomics/issues/903)); for example, if using conda, this can be set up by running `conda install python=3.9`.
2. Next, please run `pip3 install -r requirements.txt` to install the required packages.
3. Next, clone various necessary repositories (for example, PyRadiomics) by running `bash install.sh`.
4. Finally, RadFID requires the RadImageNet weights for the InceptionV3 model, which can be downloaded from RadImageNet's official source [here](https://drive.google.com/file/d/1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR/view). Once downloaded, please place the `InceptionV3.pt` checkpoint file into `gan-metrics-pytorch/models` and rename it to `RadImageNet_InceptionV3.pt`. Our code will take care of the rest.

## 1. Basic Metric Computation

You can compute all distance metrics between two sets of images using the following command:

```bash
bash compute_allmetrics.sh $IMAGE_FOLDER1 $IMAGE_FOLDER2
```

where `$IMAGE_FOLDER1` and `$IMAGE_FOLDER2` are the paths to the two folders containing the images you want to compare. This will print out the computed distances to the terminal. For example, this can be used to evaluate the performance of a generative model by comparing the generated images to a set of real reference images.

## 2. Further Evaluations: Intrinsic

### 2.1 Sample Efficiency and Computation Speed Analysis

As in our [paper](https://arxiv.org/abs/2412.01496) (Secs. 5.2 and 5.3), you can also evaluate how distance estimations and computation times change with the sample size of images used to compute the distance metrics. This can be done by running the `run_sample_efficiency.sh` script, with the same arguments as `compute_allmetrics.sh` (see [Basic Metric Computation](#basic-metric-computation)), except now, you'll need to specify the sample sizes you want to use, provided as a single string with spaces separating each size. For example, to compute the distances for sample sizes of 10, 100, 500 and 1000 images, you can run:

```bash
bash run_sample_efficiency.sh $IMAGE_FOLDER1 $IMAGE_FOLDER2 "10 100 500 1000"
```

The distance values and computation times will be printed to the terminal.

### 2.2 Sensitivity to Image Transformations

To evaluate the sensitivity of the distance metrics to image transformations (as in Sec. 5.4 of our [paper](https://arxiv.org/abs/2412.01496)), you can use the `transform_images.py` script. This script applies a set of transformations to a folder of images `$IMAGE_FOLDER` and saves the transformed images in separate folders. The transformations include Gaussian blur and sharpness adjustment with different parameters (kernel sizes of 5 and 9, and sharpness factors of 0, 0.5 and 2, respectively), as well as RandomMotion from [TorchIO](https://github.com/TorchIO-project/torchio) with `degrees` and `translation` both ranging in [2,5,10]. The script can be run with the following command:

```bash
python3 transform_images.py $IMAGE_FOLDER
```

where `$IMAGE_FOLDER` is the path to the folder containing the images you want to transform. The script will create a new folder called `transformed_images` in the same directory as the input folder, and save the transformed images in subfolders named after the transformation type (e.g., `gaussian_blur`).

Transformed images for the input folder will be saved in additional folders within the same directory, one for each type of transformation. From here, the sensitivity of the distance metrics to a type of transformation can be evaluated simply by computing the distance metrics between the non-transformed image folder and the transformed image folder (see [Basic Metric Computation](#basic-metric-computation)). For example, for a transformation of Gaussian blur with kernel size 5, you can run:

```bash
bash compute_allmetrics.sh $IMAGE_FOLDER $IMAGE_FOLDER_TRANSFORMED
```

where `$IMAGE_FOLDER_TRANSFORMED` is the path to the folder containing the transformed images: `{$IMAGE_FOLDER}_gaussian_blur5` in this case.

## 3. Further Evaluations: Extrinsic

### 3.1 Correlation with Downstream Task Performance

As in Sec. 4.2 of our paper, you can evaluate the correlation between the distance metrics and the performance of a downstream task (e.g., classification, segmentation, etc.) using the `correlate_downstream_task.py` script. For example, as in our paper, this can be used to evaluate image-to-image translation models; given a test set $D_{s\rightarrow t}$ of source domain images which were translated to the target domain as well as an additional set of reference target domain images $D_t$, a distance/similarity metric $d$ (e.g., FRD) can be evaluated by seeing if $d(D_t, D_{s\rightarrow t})$ can serve as a proxy of (i.e, correlates with) the performance of some downstream task model on $D_{s\rightarrow t}$ (for example, Dice coefficient if the task is segmentation). **Note:** for this to be valid, the reference set $D_t$ must be fixed for all evaluations of $d$.

 To use this script, create a simple CSV file with the following columns:
- `distance`: the distance metric value (e.g., FRD) between the test images (e.g., generated/translated images) and the reference images
- `task_performance`: the performance of the downstream task model on the test images (e.g., Dice coefficient)

From here, you can run the script with the following command:

```bash
python3 correlate_downstream_task.py $CSV_FILE
```

where `$CSV_FILE` is the path to the CSV file you created. The script will compute the correlation between the distance metric and the downstream task performance, and print the results to the terminal.
The correlation will be computed using the Pearson linear correlation coefficient, and the Spearman and Kendall nonlinear/rank correlation coefficients; the results will be printed to the terminal. The script will also plot a scatter plot of the distance metric values against the downstream task performance, and save the plot as a PNG file in the same directory as the input CSV file.
The plot will be saved as `correlation_plot.png`, and the correlation coefficient will be printed to the terminal. The script will also print the p-value of the correlation tests, which indicates the statistical significance of the correlations.

### 3.2 Out-of-Domain/Distribution Detection

The script `ood_detection.py` allows you to evaluate the ability of different feature representations to detect out-of-distribution (OOD) images in both threshold-free and threshold-based settings, as shown in Section 4.1 of our paper. This is computed given:

1. A reference in-distribution image set (used to compute a reference feature distribution), for example, a model's training set.
2. A test set of both in-distribution (ID) and out-of-distribution (OOD) images.

This script extracts feature embeddings (e.g., standardized radiomic features as is used in FRD, or InceptionV3 features with ImageNet or RadImageNet weights as is evaluated in our paper), and evaluates:

1. Threshold-independent performance: using AUC based on distance from the ID mean.
2. Threshold-based detection: using a 95th percentile threshold on ID validation distances, to compute accuracy, TPR, TNR, and AUC.

To run this file, you can use the following command:

```bash
python3 ood_detection.py \
  --img_folder_ref_id ${IMAGE_FOLDER_REF_ID} \
  --img_folder_test_id ${IMAGE_FOLDER_TEST_ID} \
  --img_folder_test_ood ${IMAGE_FOLDER_TEST_OOD}
```

where:

- `${IMAGE_FOLDER_REF_ID}` is the path to the folder containing the reference in-distribution images.
- `${IMAGE_FOLDER_TEST_ID}` is the path to the folder containing the test in-distribution images.
- `${IMAGE_FOLDER_TEST_OOD}` is the path to the folder containing the test out-of-distribution images.

The various results will be printed to the terminal.
