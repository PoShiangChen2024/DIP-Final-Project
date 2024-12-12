import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

class ResUNet(nn.Module):
    def __init__(self, num_class=100):
        super().__init__()
         
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        self.encoder1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  
        self.encoder2 = resnet.layer1  
        self.encoder3 = resnet.layer2  
        self.encoder4 = resnet.layer3  
        self.encoder5 = resnet.layer4  

        self.up1 = up_layer_with_attention(2048, 1024, 1024)  
        self.up2 = up_layer_with_attention(1024, 512, 512)    
        self.up3 = up_layer_with_attention(512, 256, 256)    
        self.up4 = up_layer_with_attention(256, 64, 64)    

        self.last_conv = nn.Conv2d(64, num_class, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        # Decoder
        x4_up = self.up1(x5, x4)
        x3_up = self.up2(x4_up, x3)
        x2_up = self.up3(x3_up, x2)
        x1_up = self.up4(x2_up, x1)

        output = self.last_conv(x1_up)

        output = F.interpolate(output, size=(256, 256), mode="bilinear", align_corners=False)
        return output


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_prob=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob),  
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_prob)  
        )
    
    def forward(self, x):
        return self.conv(x)


class up_layer_with_attention(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, dropout_prob=0.5):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) 
        self.conv = double_conv(out_channels + skip_channels, out_channels, dropout_prob)  
        self.attention = SEBlock(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)  
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False) 
        x = torch.cat([x2, x1], dim=1) 
        x = self.conv(x)  
        x = self.attention(x) 
        return x
    


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        batch, channels, height, width = x.size()
        squeeze = x.view(batch, channels, -1).mean(dim=2)
        excitation = self.fc1(squeeze)
        excitation = F.relu(excitation, inplace=True)
        excitation = self.fc2(excitation)
        excitation = torch.sigmoid(excitation)
        excitation = excitation.view(batch, channels, 1, 1)
        return x * excitation
