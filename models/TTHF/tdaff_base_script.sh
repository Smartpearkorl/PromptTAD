#python3 ./main.py \
#        --evaluate \
#        --batch_size 128 \
#        --dataset DADA \
#        --gpu_num 0 \
#        --height 224 \
#        --width 224 \
#        --normal_class 1 \
#        --eval_every 1000 \
#        --base_model 'RN50' \
#        --general \
#        --fg \
#        --hf \
#        --aafm \
#        --other_method 'TDAFF_BASE' \
#        --exp_name 'TDAFF_BASE_RN50'

python3 ./main.py \
        --evaluate \
        --batch_size 128 \
        --dataset DoTA \
        --gpu_num 0 \
        --height 224 \
        --width 224 \
        --normal_class 1 \
        --eval_every 1000 \
        --base_model 'RN50' \
        --general \
        --fg \
        --hf \
        --aafm \
        --other_method 'TDAFF_BASE' \
        --exp_name 'TDAFF_BASE_RN50'


#python3 ./main.py \
#        --train \
#        --lr_clip 5e-6 \
#        --wd 1e-4 \
#        --epochs 15 \
#        --batch_size 128 \
#        --dataset DoTA \
#        --gpu_num 0 \
#        --height 224 \
#        --width 224 \
#        --normal_class 1 \
#        --eval_every 1000 \
#        --base_model 'RN50' \
#        --general \
#        --fg \
#        --hf \
#        --aafm \
#        --other_method 'TDAFF_BASE' \
#        --exp_name 'TDAFF_BASE_RN50_test'

#python3 ./main.py \
#        --train \
#        --lr_clip 5e-6 \
#        --wd 1e-4 \
#        --epochs 15 \
#        --batch_size 6 \
#        --dataset DoTA \
#        --gpu_num 0 \
#        --height 448 \
#        --width 448 \
#        --normal_class 1 \
#        --eval_every 10000 \
#        --base_model 'RN50x64' \
#        --general \
#        --fg \
#        --hf \
#        --aafm \
#        --other_method 'TDAFF_BASE' \
#        --exp_name 'TDAFF_BASE_RN50x64'

#python3 ./main.py \
#        --train \
#        --lr_clip 5e-6 \
#        --wd 1e-4 \
#        --epochs 15 \
#        --batch_size 128 \
#        --dataset DoTA \
#        --gpu_num 0 \
#        --height 224 \
#        --width 224 \
#        --normal_class 1 \
#        --eval_every 1000 \
#        --base_model 'ViT-B-32' \
#        --general \
#        --fg \
#        --hf \
#        --aafm \
#        --other_method 'TDAFF_BASE' \
#        --exp_name 'TDAFF_BASE_ViT-B-32'

#python3 ./main.py \
#        --train \
#        --lr_clip 5e-6 \
#        --wd 1e-4 \
#        --epochs 15 \
#        --batch_size 16 \
#        --dataset DoTA \
#        --gpu_num 0 \
#        --height 224 \
#        --width 224 \
#        --normal_class 1 \
#        --eval_every 10000 \
#        --base_model 'ViT-L-14' \
#        --general \
#        --fg \
#        --hf \
#        --aafm \
#        --other_method 'TDAFF_BASE' \
#        --exp_name 'TDAFF_BASE_ViT-L-14'