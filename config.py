LABELS_DIR='/home/guest/data_ssd/bdd/seg/labels/train/'
IMGS_DIR='/home/guest/data_ssd/bdd/seg/images/train/'
N_CLASS=1
RANDOM_SEED=1
TRAIN_BATCH_SIZE=64
VAL_BATCH_SIZE=4
TEST_BATCH_SIZE=1
MAX_EPOCHS=32
GPUS=[1] #[1] # use id 1 gpu
CHECK_VAL_EVERY_N_EPOCHS=1
TARGET_CLASS_ID=13 # CAR
CENTER_CROP_SIZE=512
DEVICE='cuda:1'
SMP_ENCODER='resnet34'
EARLY_STOPPING_PATIENCE=16
SAMPLE=False
# RESUME_FROM_CHECKPOINT='./checkpoints/epoch=9-step=1579.ckpt'
RESUME_FROM_CHECKPOINT=None
LEARNING_RATE=0.0001
IMG_WIDTH=1280
IMG_HEIGHT=720