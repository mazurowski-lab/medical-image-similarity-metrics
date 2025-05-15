#IMAGE_FOLDER1=data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/GE
#IMAGE_FOLDER2=data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/Siemens
#IMAGE_FOLDER2=data/dbc/prior_work_1k/harmonized/by_cyclegan/SiemenstoGE
#IMAGE_FOLDER1=data/brats/testB
#IMAGE_FOLDER2=data/brats/testA
#IMAGE_FOLDER2=data/brats/harmonized/t1tot2/by_gcgan
#IMAGE_FOLDER2=data/brats/harmonized/t1tot2/by_gcgan_gaussian_blur9
#IMAGE_FOLDER1=data/lumbar_mritoct/CT/images/test
#IMAGE_FOLDER2=data/lumbar_mritoct/MRI/images/test
#IMAGE_FOLDER2=data/lumbar_mritoct/harmonized/by_cut
#IMAGE_FOLDER2=data/lumbar_mritoct/harmonized/by_cut_gaussian_blur9
IMAGE_FOLDER1=data/chaos/split_by_domain/images/testB
#IMAGE_FOLDER2=data/chaos/split_by_domain/images/trainA
IMAGE_FOLDER2=data/chaos/harmonized/CTtoT1InPhaseMRI/by_unsb

python3 fid_score.py \
	--true "../../${IMAGE_FOLDER1}" \
	--fake "../../${IMAGE_FOLDER2}" \
