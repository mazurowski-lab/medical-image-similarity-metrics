# NOTE: can compute on RadImageNet features by changing line 62 of gan-metrics-pytorch/models/inception.py
cd gan-metrics-pytorch
python3 kid_score.py \
	--true ../../data/other_data/for_contourdiff/lumbar_foreground/MRI \
	--fake ../../data/other_data/for_contourdiff/lumbar_foreground/gen_FGDM200 \
	--img-size 256
cd ..

