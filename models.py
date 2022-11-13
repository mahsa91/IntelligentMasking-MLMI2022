from torch import nn
import torch
from torch import nn
from torch.functional import F
import torchmetrics
import torchvision
import pytorch_lightning as pl
from neptune.new.types import File
from utils import accuracy, dice_score, correct_list, get_figure
from matplotlib import pyplot as plt
import random


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, mode='encoder'):
        super().__init__()
        assert mode in ['encoder', 'decoder']
        if mode == 'encoder':
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        elif mode == 'decoder':
            self.conv1 = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.conv2(self.relu(self.bn(self.conv1(x))))


class Encoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.last_channel = chs[-1]
        self.chs = chs
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        for block in self.enc_blocks:
            z = block(x)
            x = self.pool(z)
        return z


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64, 32, 16)):
        super().__init__()
        self.chs = chs
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i + 1], mode='decoder') for i in range(len(chs) - 1)])

    def forward(self, x):
        for i in range(len(self.chs) - 1):
            x = self.dec_blocks[i](x)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding='same')
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding='same')

    def forward(self, x):
        return self.conv2(self.relu(self.bn(self.conv1(x))))


class UNetEncoder(nn.Module):
    def __init__(self, chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.last_channel = chs[-1]
        self.chs = chs
        self.enc_blocks = nn.ModuleList([UNetBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, return_ftrs=False):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs if return_ftrs else ftrs[::-1][0]
        


class UNetDecoder(nn.Module):
    def __init__(self, dec_chs=(1024, 512, 256, 128, 64), enc_chs=(1, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.last_channel = dec_chs[-1]
        self.dec_chs = dec_chs
        self.enc_chs_rev = enc_chs[::-1]
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(dec_chs[i], dec_chs[i + 1], 2, 2) for i in range(len(dec_chs) - 1)])
        self.dec_blocks = nn.ModuleList([UNetBlock(dec_chs[i+1] + self.enc_chs_rev[i+1], dec_chs[i + 1]) for i in range(len(dec_chs) - 1)])

    def forward(self, x, encoder_features):
        for i in range(len(self.dec_chs) - 1):
            x = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x = torch.cat([x, enc_ftrs], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class AE_UNet(nn.Module):
    def __init__(self, enc_chs=(1, 16, 32, 64, 64, 128, 128), dec_chs=(128, 128, 64, 64, 32, 16),
                 num_class=1, retain_dim=False, image_size=300, unet_flag=False):
        super().__init__()
        self.unet_flag = unet_flag
        if not unet_flag:
            self.encoder = Encoder(enc_chs)
            self.decoder = Decoder(dec_chs)
        else:
            self.encoder = UNetEncoder(enc_chs)
            self.decoder = UNetDecoder(dec_chs, enc_chs)
            
        kernel_size = 2 * (1 + int(image_size / (2 ** len(enc_chs)))) - 1
        self.channel_wise_conv = nn.Conv2d(enc_chs[-1], enc_chs[-1], kernel_size, padding='same', groups=enc_chs[-1])
        self.head = nn.Conv2d(dec_chs[-1], num_class, 1)
        self.retain_dim = retain_dim
        self.image_size = image_size

    def forward(self, x):
        if not self.unet_flag:
            z = self.extract_features(x)
            out = self.decoder(z)
        else:
            z, enc_ftrs = self.extract_features(x)
            out = self.decoder(z, enc_ftrs[::-1][1:])
        out = self.head(out)
        if self.retain_dim:
            out = F.interpolate(out, (self.image_size, self.image_size))
        return out

    def extract_features(self, x):
        if not self.unet_flag:
            z = self.encoder(x)
            z = self.channel_wise_conv(z)
            return z
        else:
            enc_ftrs = self.encoder(x, return_ftrs=True)
            z = self.channel_wise_conv(enc_ftrs[::-1][0])
            return z, enc_ftrs

    def get_embedding(self, x):
        if not self.unet_flag:
            z = self.extract_features(x)
        else:
            z, _ = self.extract_features(x)
        return F.adaptive_avg_pool2d(z, (1, 1))
    

class MaskerModelRL(nn.Module):
    def __init__(self, model_config, device='cuda'):
        super().__init__()
        self.model = AE_UNet(**model_config)
        self.image_size = model_config['image_size']
                
    def forward(self, x):
        mask = self.model.forward(x)
        return mask
    
    def get_image_size_mask(self, mask):
        image_size_mask = F.interpolate(mask, (self.image_size, self.image_size))
        return image_size_mask

    def create_mask(self, image, mask_size, current_epoch, max_epochs, mask_value=1.0):
        if isinstance(mask_size, int):
            mask_h = mask_size
            mask_w = mask_size
        else:
            mask_h = mask_size[0]
            mask_w = mask_size[1]

        image_h = image.shape[-2]
        image_w = image.shape[-1]
        
        mask = torch.zeros_like(image, dtype=torch.float)
        mask_original = torch.zeros_like(image, dtype=torch.float)

        assert mask_h <= image_h and mask_w <= image_w

        for i in range(len(image)):
            # threshold = 1 - 3*current_epoch/max_epochs
            # if random.random() < threshold:
            if current_epoch < max_epochs * 0.2:
                mask_start_h = random.randint(0, image_h - mask_h)
                mask_start_w = random.randint(0, image_w - mask_w)
            else:
                mask_start_h, mask_start_w = (image[i][0]==torch.max(image[i][0])).nonzero()[0]
            mask_original[i, 0][mask_start_h, mask_start_w] = mask_value
            mask[i, 0][mask_start_h:mask_start_h + mask_h, mask_start_w:mask_start_w + mask_w] = mask_value
            
        return mask, mask_original

    
class IntelligentMaskModelRL(pl.LightningModule):
    def __init__(self, recon_model_config, masker_model_config, lr_masker_model=1e-3, lr_recon_model=1e-3, run_id=-1, loss_on_mask=False, use_scheduler=True, milestones=[100, 200]):
        super().__init__()
        self.save_hyperparameters()

        self.recon_model = AE_UNet(**recon_model_config)
        self.masker_model = MaskerModelRL(masker_model_config)
        self.lr_masker_model = lr_masker_model
        self.lr_recon_model = lr_recon_model
        
        self.loss = nn.MSELoss(reduction='none')
        self.loss_on_mask = loss_on_mask
        self.use_scheduler = use_scheduler
        self.milestones = milestones
        self.run_id = run_id
        self.automatic_optimization = False

    def forward(self, x):
        return self.model.forward(x)

    def get_loss(self, x_pred, x, mask):
        if self.loss_on_mask:
            loss = torch.sum(self.loss(x_pred, x) * mask) / torch.sum(mask)
        else:
            loss = torch.mean(self.loss(x_pred, x))
        return loss
        
    def training_step(self, train_batch, batch_idx):
        opt_recon, opt_masker = self.optimizers()
        scheduler_recon, scheduler_masker = self.lr_schedulers()
        
        x, _ = train_batch
        loss_pred = self.masker_model(x)
        mask, mask_original = self.masker_model.create_mask(loss_pred, 2, self.current_epoch, self.trainer.max_epochs)
        mask = self.masker_model.get_image_size_mask(mask)
        x_masked = (1 - mask) * x 
        
        x_pred = self.recon_model.forward(x_masked)

        # Train Reconstruction model
        opt_recon.zero_grad()
        recon_loss = self.get_loss(x_pred, x, mask)
        self.log('train_loss', recon_loss, prog_bar=True)
        self.manual_backward(recon_loss, retain_graph=True)
        opt_recon.step()
        if self.use_scheduler and self.trainer.is_last_batch:
            scheduler_recon.step()
        
        # Train Masker model
        opt_masker.zero_grad()
        x_pred = x_pred.detach()
        recon_loss_per_batch = torch.sum(self.loss(x_pred, x.detach()), dim=(1,2,3))
        loss_pred_per_batch = torch.sum(mask_original * loss_pred, dim=(1,2,3))
        masker_loss = torch.mean(self.loss(recon_loss_per_batch, loss_pred_per_batch))
        self.manual_backward(masker_loss)
        opt_masker.step()
        if self.use_scheduler and self.trainer.is_last_batch:
            scheduler_masker.step()

        # Log image sample
        if batch_idx == 0:
            fig = get_figure(x[0], loss_pred[0], mask[0], x_masked[0], x_pred[0])
            plt.title(f'True Loss:{recon_loss_per_batch[0]:.2f}, Pred Loss: {loss_pred_per_batch[0]:.2f}')
            plt.show()
            plt.close()


    def on_before_zero_grad(self, optimizer):
        parameters = [p for p in self.masker_model.parameters() if p.grad is not None]
        if len(parameters) > 0:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in parameters])).item()
        else:
            total_norm = -1
        self.log("total_grad_norm", total_norm)

    def validation_step(self, val_batch, batch_idx):
        x, _ = val_batch
        loss_pred = self.masker_model(x)
        mask, mask_original = self.masker_model.create_mask(loss_pred, 3, self.current_epoch, self.trainer.max_epochs)
        mask = self.masker_model.get_image_size_mask(mask)
        x_masked = (1 - mask) * x
        x_pred = self.recon_model.forward(x_masked)
        loss = self.get_loss(x_pred, x, mask)
        self.log('val_loss', loss, prog_bar=True)
        
    def configure_optimizers(self):
        opt_recon_model = torch.optim.Adam(self.recon_model.parameters(), lr=self.lr_recon_model)
        opt_masker_model = torch.optim.Adam(self.masker_model.parameters(), lr=self.lr_masker_model)
        scheduler_recon = torch.optim.lr_scheduler.MultiStepLR(opt_recon_model, milestones=self.milestones, gamma=0.33)
        scheduler_masker = torch.optim.lr_scheduler.MultiStepLR(opt_masker_model, milestones=self.milestones, gamma=0.33)
        return [opt_recon_model, opt_masker_model], [scheduler_recon, scheduler_masker]


class ClfBlock(nn.Module):
    def __init__(self, nfeat, dropout, hidden=-1, nclass=1):
        super().__init__()
        layers = []
        if hidden == -1 or len(hidden) == 0:
            layers.append(nn.Linear(nfeat, nclass))
        else:
            layers.append(nn.Linear(nfeat, hidden[0]))
            for i in range(len(hidden) - 1):
                layers.append(nn.Linear(hidden[i], hidden[i + 1]))
            layers.append(nn.Linear(hidden[-1], nclass))
        self.clflayers = nn.ModuleList(layers)
        self.dropout = dropout

    def forward(self, x):
        end_layer = len(self.clflayers) - 1
        for i in range(end_layer):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.clflayers[i](x)
            x = F.relu(x)

        x = F.dropout(x, self.dropout, training=self.training)
        output = self.clflayers[-1](x)
        output = F.log_softmax(output, dim=1)
        return output


class BaseClassifier(pl.LightningModule):
    def __init__(self, nclass):
        super().__init__()
        self.lr = None
        self.weight_decay = None
        self.use_scheduler = None
        self.milestones = None
        self.encoder = None
        self.clf = None
        self.loss = None

        self.train_acc = torchmetrics.Accuracy() 
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        self.test_in_val_acc = torchmetrics.Accuracy()

        self.train_f1mac = torchmetrics.F1Score(nclass,average='macro') 
        self.val_f1mac = torchmetrics.F1Score(nclass,average='macro')
        self.test_f1mac = torchmetrics.F1Score(nclass,average='macro')
        self.test_in_val_f1mac = torchmetrics.F1Score(nclass,average='macro')

        self.train_auroc = torchmetrics.AUROC(nclass, average='macro') 
        self.val_auroc = torchmetrics.AUROC(nclass, average='macro')
        self.test_auroc = torchmetrics.AUROC(nclass, average='macro')
        self.test_in_val_auroc = torchmetrics.AUROC(nclass, average='macro')

        self.test_confmat = torchmetrics.ConfusionMatrix(num_classes=nclass)

    def forward(self, x):
        z = self.extract_features(x)
        pred = self.clf(z)
        return pred

    def extract_features(self, x):
        z = self.encoder(x)
        return z

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.use_scheduler:
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=30)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.33)
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss/dataloader_idx_0"}}
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log('train_loss', loss.item(), prog_bar=True, on_step=False, on_epoch=True, logger=True)

        self.train_acc(preds, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)

        self.train_f1mac(preds, y)
        self.log('train_f1macro', self.train_f1mac, on_step=False, on_epoch=True)

        self.train_auroc(preds, y)
        self.log('train_auroc', self.train_auroc, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx):
        x, y = val_batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        if dataloader_idx == 0:
            self.log('val_loss', loss, prog_bar=True)
            self.val_acc(preds, y)
            self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

            self.val_f1mac(preds, y)
            self.log('val_f1macro', self.val_f1mac, on_step=False, on_epoch=True)

            self.val_auroc(preds, y)
            self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True)

        if dataloader_idx == 1:
            self.log('test_in_val_loss', loss, prog_bar=True)
            self.test_in_val_acc(preds, y)
            self.log('test_in_val_acc', self.test_in_val_acc, on_step=False, on_epoch=True)

            self.test_in_val_f1mac(preds, y)
            self.log('test_in_val_f1macro', self.test_in_val_f1mac, on_step=False, on_epoch=True)

            self.test_in_val_auroc(preds, y)
            self.log('test_in_val_auroc', self.test_in_val_auroc, on_step=False, on_epoch=True)


    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        self.log('test_loss', loss, prog_bar=True)
        self.test_acc(preds, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        self.test_f1mac(preds, y)
        self.log('test_f1macro', self.test_f1mac, on_step=False, on_epoch=True)

        self.test_auroc(preds, y)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True)

        torch.use_deterministic_algorithms(False)
        self.test_confmat(preds, y)
        torch.use_deterministic_algorithms(True)

    def test_epoch_end(self, outs):
        self.logger.experiment['training/test_confmat'].log(str(self.test_confmat.compute())[7:-1])



class NNClassifier(BaseClassifier):
    def __init__(self, nfeat, nclass, hidden=-1, lr=5e-3, weight_decay=0, dropout=0.5, use_scheduler=True, milestones=[50, 100, 150], use_weight=False, weight=None):
        super().__init__(nclass=nclass)
        self.save_hyperparameters()
        self.encoder = nn.Identity()
        self.clf = ClfBlock(nfeat=nfeat, hidden=hidden, nclass=nclass, dropout=dropout)
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_scheduler = use_scheduler
        self.milestones = milestones
        if use_weight:
            class_weights = 10 * (1 - weight/weight.sum())
            self.loss = nn.NLLLoss(weight=class_weights)
        else:
            self.loss = nn.NLLLoss()


class CNNClassifier(NNClassifier):
    def __init__(self, enc_chs=(1, 64, 128, 256, 512, 1024), conv_channel=-1, nclass=1, hidden=-1, lr=1e-3, weight_decay=0, dropout=0.5, use_scheduler=True, milestones=[50, 100, 150], use_weight=False, weight=None):
        super().__init__(nfeat=conv_channel, nclass=nclass, hidden=hidden, lr=lr, weight_decay=weight_decay, dropout=dropout, use_scheduler=use_scheduler, milestones=milestones, use_weight=use_weight, weight=weight)
        self.save_hyperparameters()
        self.encoder = nn.Sequential(Encoder(enc_chs), nn.Conv2d(enc_chs[-1], conv_channel, 3, padding='same'), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())


class ResNetClassifier(NNClassifier):
    def __init__(self, nclass=1, hidden=-1, lr=1e-3, weight_decay=0, dropout=0.5, nlayer_unfreeze=-1, use_scheduler=True, milestones=[50, 100, 150], use_weight=False, weight=None):
        super().__init__(nfeat=torchvision.models.resnet18(pretrained=True).fc.in_features, nclass=nclass, hidden=hidden, dropout=dropout, use_scheduler=use_scheduler, milestones=milestones, use_weight=use_weight, weight=weight)
        self.save_hyperparameters()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        set_parameter_requires_grad_layered(self.resnet, nlayer_unfreeze)
        self.encoder = nn.Sequential(self.resnet, nn.Flatten())


class EncoderClassifier(NNClassifier):
    def __init__(self, encoder, encoder_last_channel, conv_channel=-1, nclass=1, hidden=-1, lr=1e-3, weight_decay=0, dropout=0.5, nlayer_unfreeze=-1, use_scheduler=True, milestones=[50, 100, 150], use_weight=False, weight=None):
        super().__init__(nfeat=conv_channel, nclass=nclass, hidden=hidden, lr=lr, dropout=dropout, use_scheduler=use_scheduler, milestones=milestones, use_weight=use_weight, weight=weight)
        self.save_hyperparameters()
        set_parameter_requires_grad_layered(encoder, nlayer_unfreeze)
        self.encoder = nn.Sequential(encoder, nn.Conv2d(encoder_last_channel, conv_channel, 3, padding='same'), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())



def set_parameter_requires_grad_layered(model, nlayer_unfreeze):
    assert (nlayer_unfreeze == 'all' or isinstance(nlayer_unfreeze, int))
    if nlayer_unfreeze != 'all':
        all_modules =  [module for module in model.modules() if \
                        len(list(module.children()))==0 and len(list(module.parameters()))>0]
        for module in all_modules[:len(all_modules) - nlayer_unfreeze]:
            print(f'Disable Grad for: {module}')
            for param in module.parameters():
                param.requires_grad = False
