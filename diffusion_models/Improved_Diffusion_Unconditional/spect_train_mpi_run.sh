export CUDA_VISIBLE_DEVICES=0

MODEL_FLAGS='--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma False --dropout 0.3'
DIFFUSION_FLAGS='--diffusion_steps 200 --noise_schedule linear'
TRAIN_FLAGS='--lr 1e-4 --batch_size 2'
SAVE_FLAGS='--save_dir /data/nas07/PersonalData/PersonalData/apoman123/wavepurifier_sc09_200_32 --log_interval 10 --save_interval 2000'
DATA_DIR="/data/nas07/SharedBySMB/apoman123/sc09"
NUM_GPUS=1
nohup \
mpiexec --allow-run-as-root --oversubscribe -n $NUM_GPUS python spectrogram_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS $SAVE_FLAGS \
> diffusion_steps=200.log&