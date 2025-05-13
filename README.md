# Medical Image 

#### By [Nicholas Konz](https://nickk124.github.io/), [Richard Osuala](https://scholar.google.com/citations?user=0KkVRVQAAAAJ&hl=en), other authors...

arXiv paper link: [![arXiv Paper](https://img.shields.io/badge/arXiv-2412.01496-orange.svg?style=flat)](https://arxiv.org/abs/2412.01496)

<p align="center">
  <img src='https://github.com/mazurowski-lab/medical-image-similarity-metrics/blob/main/figs/evalframework.png' width='95%'>
</p>

We provide an easy-to-use framework for evaluating distance/similarity metrics between unpaired sets of medical images with a variety of metrics, accompanying our [paper](https://arxiv.org/abs/2412.01496). For example, this can be used to evaluate the performance of image generative models in the medical imaging domain. The codebase includes implementations of several distance metrics that can be used to compare images, as well as tools for evaluating the performance of generative models on various downstream tasks.

Included metrics:
1. [FRD](https://arxiv.org/abs/2412.01496) (Fréchet Radiomic Distance)
2. [FID](https://papers.nips.cc/paper_files/paper/2017/hash/8a1d694707eb0fefe65871369074926d-Abstract.html) (Fréchet Inception Distance)
3. [Radiology FID/RadFID](https://www.spiedigitallibrary.org/journals/journal-of-medical-imaging/volume-10/issue-06/061403/medigan--a-Python-library-of-pretrained-generative-models-for/10.1117/1.JMI.10.6.061403.full)
4. [KID](https://openreview.net/forum?id=r1lUOzWCW) (Kernel Inception Distance)
5. [CMMD](https://arxiv.org/abs/2401.09603) (CLIP Maximum Mean Discrepancy)

## Credits

Thanks to the following repositories which this framework builds upon:
1. [frd-score](https://github.com/RichardObi/frd-score)
2. [gan-metrics-pytorch](https://github.com/abdulfatir/gan-metrics-pytorch)
3. [cmmd-pytorch](https://github.com/sayakpaul/cmmd-pytorch)

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

## Installation/Setup

1. First, note that Python <=3.9 is required due to one of the distances, FRD, using PyRadiomics (see [here](https://github.com/AIM-Harvard/pyradiomics/issues/903)); for example, if using conda, this can be set up by running `conda install python=3.9`.
2. Next, please run `pip3 install -r requirements.txt` to install the required packages.
<!-- 3. Finally, install PyRadiomics by running `bash install.sh`. ???-->

## Usage

### Basic Metric Computation

You can compute all distance metrics between two sets of images using the following command:

```bash
bash compute_allmetrics.sh $IMAGE_FOLDER1 $IMAGE_FOLDER2
```

where `$IMAGE_FOLDER1` and `$IMAGE_FOLDER2` are the paths to the two folders containing the images you want to compare. This will print out the computed distances to the terminal. For example, this can be used to evaluate the performance of a generative model by comparing the generated images to a set of real reference images.

### Further Evaluations

#### Sample Efficiency and Computation Speed Analysis

As in our [paper](https://arxiv.org/abs/2412.01496) (Secs. 5.2 and 5.3), you can also evaluate how distance estimations and computation times change with the sample size of images used to compute the distance metrics. This can be done by running the `run_sample_efficiency.sh` script, with the same arguments as `compute_allmetrics.sh` (see [Basic Metric Computation](#basic-metric-computation)), except now, you'll need to specify the sample sizes you want to use, provided as a single string with spaces separating each size. For example, to compute the distances for sample sizes of 10, 100, 500 and 1000 images, you can run:

```bash
bash run_sample_efficiency.sh $IMAGE_FOLDER1 $IMAGE_FOLDER2 "10 100 500 1000"
```

The distance values and computation times will be printed to the terminal.

### Sensitivity to Image Transformations

To evaluate the sensitivity of the distance metrics to image transformations (as in Sec. 5.4 of our [paper](https://arxiv.org/abs/2412.01496)), you can use the `transform_images.py` script. This script applies a set of transformations to a folder of images `$IMAGE_FOLDER` and saves the transformed images in separate folders. The transformations include Gaussian blur and sharpness adjustment with different parameters (kernel sizes of 5 and 9, and sharpness factors of 0, 0.5 and 2, respectively). The script can be run with the following command:

```bash
python3 transform_images.py $IMAGE_FOLDER
```

where `$IMAGE_FOLDER` is the path to the folder containing the images you want to transform. The script will create a new folder called `transformed_images` in the same directory as the input folder, and save the transformed images in subfolders named after the transformation type (e.g., `gaussian_blur`, `sharpness_adjustment`).

Transformed images for the input folder will be saved in additional folders within the same directory, one for each type of transformation. From here, the sensitivity of the distance metrics to a type of transformation can be evaluated simply by computing the distance metrics between the non-transformed image folder and the transformed image folder (see [Basic Metric Computation](#basic-metric-computation)). For example, for a transformation of Gaussian blur with kernel size 5, you can run:

```bash
bash compute_allmetrics.sh $IMAGE_FOLDER $IMAGE_FOLDER_TRANSFORMED
```

where `$IMAGE_FOLDER_TRANSFORMED` is the path to the folder containing the transformed images: `{$IMAGE_FOLDER}_gaussian_blur5` in this case.