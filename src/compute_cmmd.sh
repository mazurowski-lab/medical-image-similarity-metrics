cd cmmd-pytorch
python3 main_cmmd.py \
	../../data/lumbar_mritoct/CT/images/test \
	../../data/lumbar_mritoct/harmonized/by_cut \
       	--batch_size=32 \
	--max_count=30000
cd ..
