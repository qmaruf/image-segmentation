import dataset
import models
import config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import setproctitle
import argparse
from torchvision import transforms as tvt
import torch
import torch.nn as nn
import cv2


op_sigmoid = nn.Sigmoid()

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-phase','--phase', help='train/predict', required=True, choices=['train', 'test'])
parser.add_argument('-model','--model', help='model path', required=False)
args = vars(parser.parse_args())

print (args)
setproctitle.setproctitle('__maruf__')

checkpoint_callback = ModelCheckpoint(
    filepath='./checkpoints/',
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.01,
   patience=config.EARLY_STOPPING_PATIENCE,
   verbose=False,
   mode='min',
)

def predict():
    train_dl, val_dl, test_dl = dataset.get_data_loader()
    model = models.SemanticSegmentationModel().to(config.DEVICE)
    model.load_from_checkpoint(args['model'])
    model.eval()
    n = 0
    with torch.no_grad():
        for batch in train_dl:
            images, masks = batch['image'], batch['mask']
            images = images.to(config.DEVICE)
            predictions = model(images)
            predictions = (op_sigmoid(predictions) > 0.5).type(torch.int)
            prediction = predictions[0][0].data.cpu().numpy()*255        
            cv2.imwrite('prediction%d.jpg'%n, prediction)
            n += 1
        # exit()
        
def run():
    train_dl, val_dl, test_dl = dataset.get_data_loader()
    model = models.SemanticSegmentationModel(train_dl, val_dl, test_dl)
    trainer = pl.Trainer(max_epochs=config.MAX_EPOCHS,
                        gpus=config.GPUS, 
                        check_val_every_n_epoch=config.CHECK_VAL_EVERY_N_EPOCHS,
                        auto_lr_find=True,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        resume_from_checkpoint=config.RESUME_FROM_CHECKPOINT)    
    # trainer.tune(model)
    trainer.fit(model)

if __name__ == '__main__':
    if args['phase'] == 'train':
        run()
    else:
        if args['model']:
            predict()