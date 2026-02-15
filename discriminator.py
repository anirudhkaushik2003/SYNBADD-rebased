# =============================================================================
# File: discriminator.py
# Purpose: Discriminator model definitions and helper utilities used in adversarial training variants.
# Prerequisites: Imported by scripts enabling --use_discriminator.
# Usually Run Next: Used by train_VAE_SYNBADD_svf.py and compatible test/eval scripts when discriminator checkpoints are present.
# Model Components Used: Defines DAFTDiscriminator and legacy discriminator variants.
# =============================================================================

import torch.nn as nn
import torch
from daft_blocks import DAFTBlock

def weights_init_normal(model):
    '''
    More of a stable init for this problem than the default Pytorch init
    :param model: the model to initialise
    :return:
    '''
    classname = model.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)


def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
    if maxpool is True:
        layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.MaxPool3d(2, stride=maxpool_stride),
            nn.ReLU(),
        )
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
            nn.BatchNorm3d(out_channel),
            nn.ReLU()
        )
    return layer

class DAFTDiscriminator(nn.Module):
    def __init__(self, in_channels, ndim_non_img, channel_number=[32, 64, 128, 256, 64]):
        super(DAFTDiscriminator, self).__init__()
        n_layer = len(channel_number)
        
        # Instead of a sequential module, we'll use ModuleList to allow DAFT blocks
        self.layers = nn.ModuleList()
        
        # DAFT blocks for conditioning - one for each layer except the last one
        self.daft_blocks = nn.ModuleList()
        
        for i in range(n_layer):
            if i == 0:
                in_channel = in_channels
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            
            if i < n_layer - 1:
                # Add regular convolutional layer
                self.layers.append(
                    nn.Sequential(
                        nn.Conv3d(in_channel, out_channel, padding=1, kernel_size=3),
                        nn.BatchNorm3d(out_channel),
                        nn.MaxPool3d(2, stride=2),
                        nn.ReLU(),
                    )
                )
                
                # Add corresponding DAFT block for this layer
                self.daft_blocks.append(
                    DAFTBlock(
                        in_channels=out_channel,
                        out_channels=out_channel,
                        bn_momentum=0.1,
                        stride=1,  # No stride in DAFT block as we already have MaxPool
                        ndim_non_img=ndim_non_img,
                        location=4,  # After ReLU - you can change this as needed
                        activation="linear",
                        scale=True,
                        shift=True,
                        bottleneck_dim=7,
                    )
                )
            else:
                # Last layer without DAFT
                self.layers.append(
                    nn.Sequential(
                        nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1),
                        nn.BatchNorm3d(out_channel),
                        nn.ReLU()
                    )
                )
        
        # Add adaptive average pooling to ensure correct output size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Output for adversarial task
        self.classifier_adv = nn.Sequential(
            nn.Conv3d(channel_number[-1], 1, kernel_size=1, padding=0, bias=False)
        )
    
    def forward(self, x, aux_input):
        # Pass through the layers with DAFT conditioning
        for i, (layer, daft_block) in enumerate(zip(self.layers[:-1], self.daft_blocks)):
            x = layer(x)
            x = daft_block(x, aux_input)
        
        # Process the last layer (without DAFT)
        x = self.layers[-1](x)
        
        # Apply adaptive pooling
        x = self.adaptive_pool(x)
        
        # Get adversarial predictions
        adversarial_preds = self.classifier_adv(x)
        adversarial_preds = adversarial_preds.view(adversarial_preds.size(0), -1)
        
        return adversarial_preds

class SFCNDiscriminator(nn.Module):
    def __init__(self, in_channels, channel_number=[32, 64, 128, 256, 64]):
        super(SFCNDiscriminator, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = in_channels
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(f'conv_{i}',
                                                conv_layer(in_channel,
                                                         out_channel,
                                                         maxpool=True,
                                                         kernel_size=3,
                                                         padding=1))
            else:
                self.feature_extractor.add_module(f'conv_{i}',
                                                conv_layer(in_channel,
                                                         out_channel,
                                                         maxpool=False,
                                                         kernel_size=1,
                                                         padding=0))

        in_channel = channel_number[-1]
        
        # Add adaptive average pooling to ensure correct output size
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Output for adversarial task
        self.classifier_adv = nn.Sequential(
            nn.Conv3d(in_channel, 1, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x):
        encoded_features = self.feature_extractor(x)
        
        # Apply adaptive pooling to get consistent spatial dimensions
        encoded_features = self.adaptive_pool(encoded_features)

        adversarial_preds = self.classifier_adv(encoded_features)

        adversarial_preds = adversarial_preds.view(x.size(0), -1)
        
        return adversarial_preds
