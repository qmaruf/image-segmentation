import torch
import config
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models as tvmodels
import segmentation_models_pytorch as smp
from torchvision import transforms as tvt

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class SemanticSegmentation(nn.Module):
  def __init__(self, n_classes):
    super(SemanticSegmentation, self).__init__()
    self.model = model = smp.DeepLabV3(
        encoder_name=config.SMP_ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for grayscale images, 3 for RGB, etc.)
        classes=2, 
    )   

  def forward(self, input_):
    prediction = self.model(input_)
    return prediction


def tensor_to_img(tnsr, output):
    img = tvt.ToPILImage()(tnsr)
    img.save(output)
    # exit()
    
class SemanticSegmentationModel(pl.LightningModule):
    def __init__(self, train_dl=None, val_dl=None, test_dl=None):
        super(SemanticSegmentationModel, self).__init__()        
        self.model = SemanticSegmentation(n_classes=config.N_CLASS)
        self.diceloss = smp.losses.DiceLoss(mode='binary')         
        self.learning_rate = 0.0001     
        self.train_dl, self.val_dl, self.test_dl = train_dl, val_dl, test_dl
        self.sigmoid = nn.Sigmoid()
        self.mseloss = nn.MSELoss()
        self.cnt = 0

    def criterion(self, mask_pred, mask_true):
        
        '''
        return diceloss * 0.8 + diffloss * 0.2 works good
        '''
        
        pred_segm = mask_pred[:, 0, :, :].unsqueeze(1)
        pred_fail = mask_pred[:, 1, :, :].unsqueeze(1)
        
        diceloss = self.diceloss(pred_segm, mask_true)
        
        pred_segm = self.sigmoid(pred_segm)
        diff = mask_true - pred_segm
        diff = (diff > 0.5).type(torch.float) # detect high difference area
                
        # if self.cnt % 10 == 0:
        #     tensor_to_img(diff[0], 'diff.jpg')
        #     tensor_to_img(pred_segm[0], 'prediction.jpg')
        #     tensor_to_img(mask_true[0], 'label.jpg')
        
        self.cnt = (self.cnt + 1)%10
        
        diffloss = self.diceloss(pred_fail, diff)
        
        # self.log('')
        # print ('\n diceloss \t %f diffloss \t %f'%(diceloss, diffloss))
        
        
        loss = diceloss * 0.9 + diffloss * 0.1
        
        return loss
        
        
        if self.cnt < 5:
            print ('using diceloss')
            loss = diceloss
        else:
            print ('using diffloss')
            loss = diffloss
            
        return loss
        
        
    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_nb):
        inputs, targets = batch['image'], batch['mask']
        preds = self.forward(inputs)        
        loss = self.criterion(preds, targets)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        inputs, targets = batch['image'], batch['mask']
        preds = self.forward(inputs)                
        loss = self.criterion(preds, targets)
        return {'val_loss': self.criterion(preds, targets)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        inputs, targets = batch['image'], batch['mask']
        preds = self.forward(inputs)                
        return {'test_loss': self.criterion(preds, targets)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl