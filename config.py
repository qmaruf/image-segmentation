BDD_LABELS_DIR='/home/guest/data_ssd/bdd/seg/labels/train/'
BDD_IMGS_DIR='/home/guest/data_ssd/bdd/seg/images/train/'
CARLA_LABELS_DIR='/home/guest/data_ssd/carla/seg/'
CARLA_IMGS_DIR='/home/guest/data_ssd/carla/camera/'
N_CLASS=1
RANDOM_SEED=1
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=4
TEST_BATCH_SIZE=1
MAX_EPOCHS=32
GPUS=[1] #[1] # use id 1 gpu
CHECK_VAL_EVERY_N_EPOCHS=1
BDD_TARGET_CLASS_ID=13 # CAR
CENTER_CROP_SIZE=512
DEVICE='cuda:1'
SMP_ENCODER='mobilenet_v2'
EARLY_STOPPING_PATIENCE=16
SAMPLE=False
RESUME_FROM_CHECKPOINT='./checkpoints/CARLA-epoch=20-step=377.ckpt'
# RESUME_FROM_CHECKPOINT=None
LEARNING_RATE=0.0001
BDD_IMG_WIDTH=1280
BDD_IMG_HEIGHT=720
CARLA_IMG_WIDTH=1280
CARLA_IMG_HEIGHT=720
DATASET='CARLA' # 'BDD'
CARLA_TARGET_CLASS_ID=200 # CAR



# ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
# 'vgg19', 'vgg19_bn', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 'timm-tf_efficientnet_lite4', 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e', 'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d', 'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s', 'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s', 'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040', 'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120', 'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002', 'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008', 'timm-regnety_016',
# 'timm-regnety_032', 'timm-regnety_040', 'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120', 'timm-regnety_160', 'timm-regnety_320', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d']