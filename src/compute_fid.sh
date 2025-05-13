cd gan-metrics-pytorch
python3 fid_score.py \
	--true ../../data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/GE \
	--fake ../../data/dbc/prior_work_1k/mri_data_labeled2D/split_by_domain/test/Siemens \
cd ..
