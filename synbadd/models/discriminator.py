"""DAFT-conditioned discriminator for adversarial training."""

import torch.nn as nn

from synbadd.models.daft_blocks import DAFTBlock
from synbadd.models.registry import register_model


@register_model("DAFTDiscriminator")
class DAFTDiscriminator(nn.Module):
    """Discriminator with DAFT conditioning at each convolutional layer."""

    def __init__(self, in_channels, ndim_non_img, channel_number=None):
        super().__init__()
        if channel_number is None:
            channel_number = [32, 64, 128, 256, 64]

        n_layer = len(channel_number)
        self.layers = nn.ModuleList()
        self.daft_blocks = nn.ModuleList()

        for i in range(n_layer):
            in_ch = in_channels if i == 0 else channel_number[i - 1]
            out_ch = channel_number[i]

            if i < n_layer - 1:
                self.layers.append(nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, padding=1, kernel_size=3),
                    nn.BatchNorm3d(out_ch),
                    nn.MaxPool3d(2, stride=2),
                    nn.ReLU(),
                ))
                self.daft_blocks.append(DAFTBlock(
                    in_channels=out_ch, out_channels=out_ch,
                    bn_momentum=0.1, stride=1, ndim_non_img=ndim_non_img,
                    location=4, activation="linear",
                    scale=True, shift=True, bottleneck_dim=7,
                ))
            else:
                self.layers.append(nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, padding=0, kernel_size=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(),
                ))

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier_adv = nn.Sequential(
            nn.Conv3d(channel_number[-1], 1, kernel_size=1, padding=0, bias=False)
        )

    def forward(self, x, aux_input):
        for layer, daft_block in zip(self.layers[:-1], self.daft_blocks):
            x = layer(x)
            x = daft_block(x, aux_input)

        x = self.layers[-1](x)
        x = self.adaptive_pool(x)
        adversarial_preds = self.classifier_adv(x)
        return adversarial_preds.view(adversarial_preds.size(0), -1)
