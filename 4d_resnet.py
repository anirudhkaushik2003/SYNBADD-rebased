# =============================================================================
# File: 4d_resnet.py
# Purpose: Support or experiment file within SYNBADD repository.
# Prerequisites: Depends on imports and paths defined in this file.
# Usually Run Next: Typically called by a train/test/eval script in this repo.
# Model Components Used: See imports in this file for exact model/loss dependencies.
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict

# Creating a mapping between TensorFlow model names and PyTorch model classes
model_map = {
    'Resnet4D': Resnet4D,
    'Resnet3D': Resnet3D,
    'convGRUCNN': ConvGRUCNN,
    'CNNGRU': CNNGRU,
}

def get_model(mdl_params):
    """Returns a model instance based on model_type
    Args:
        mdl_params: dictionary, contains configuration
    Returns:
        model: An instance of the desired PyTorch model
    Raises:
        ValueError: If model name is not recognized.
    """
    if mdl_params['model_type'] not in model_map:
        raise ValueError(f"Name of model unknown {mdl_params['model_type']}")
    
    model_class = model_map[mdl_params['model_type']]
    return model_class(mdl_params)


def resneta_block(x, num_fm_out, stride, factorize=False, kernel_size=3, is_temporal=False):
    """Defines a single resnetA block
    Args: 
      x: block input tensor
      num_fm_out: base number of feature maps in the block
    Returns:
      output: tensor, output of the block 
    """
    # Number of input feature maps
    num_fm_in = x.size(1)
    # Determine if its a reduction
    increase_dim = num_fm_out > num_fm_in
    
    # Create sequential layers
    layers = []
    
    # First 3x3 layer
    if factorize:
        layer = nn.Sequential(
            nn.Conv3d(num_fm_in, num_fm_out, kernel_size=[1, kernel_size, kernel_size], stride=stride, padding=[0, kernel_size//2, kernel_size//2]),
            nn.BatchNorm3d(num_fm_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_fm_out, num_fm_out, kernel_size=[kernel_size, 1, 1], stride=1, padding=[kernel_size//2, 0, 0]),
            nn.BatchNorm3d(num_fm_out),
            nn.ReLU(inplace=True)
        )
    else:
        layer = nn.Sequential(
            nn.Conv3d(num_fm_in, num_fm_out, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
            nn.BatchNorm3d(num_fm_out),
            nn.ReLU(inplace=True)
        )
    
    # Second 3x3 layer, no activation, only batch normalization
    if factorize:
        layer2 = nn.Sequential(
            nn.Conv3d(num_fm_out, num_fm_out, kernel_size=[1, kernel_size, kernel_size], stride=1, padding=[0, kernel_size//2, kernel_size//2]),
            nn.BatchNorm3d(num_fm_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_fm_out, num_fm_out, kernel_size=[kernel_size, 1, 1], stride=1, padding=[kernel_size//2, 0, 0]),
            nn.BatchNorm3d(num_fm_out)
        )
    else:
        layer2 = nn.Sequential(
            nn.Conv3d(num_fm_out, num_fm_out, kernel_size=kernel_size, stride=1, padding=kernel_size//2),
            nn.BatchNorm3d(num_fm_out)
        )
    
    # Shortcut connection
    # When the channels of input layer and conv2 do not match, we add zero pads to increase the
    # depth of input layers
    adjusted_input = x
    if stride[0] == 2 or stride[1] == 2 or stride[2] == 2:
        # Handle different dimensionality
        if x.dim() == 5:  # 3D case
            if is_temporal:
                adjusted_input = F.avg_pool3d(adjusted_input, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
            else:
                adjusted_input = F.avg_pool3d(adjusted_input, kernel_size=2, stride=2, padding=0)
        elif x.dim() == 4:  # 2D case
            adjusted_input = F.avg_pool2d(adjusted_input, kernel_size=2, stride=2, padding=0)
        else:  # 1D case
            adjusted_input = F.avg_pool1d(adjusted_input, kernel_size=2, stride=2, padding=0)
    
    # Handle the increase in dimension
    if increase_dim:
        lower_pad = math.ceil((num_fm_out-num_fm_in)/2)
        upper_pad = (num_fm_out-num_fm_in)-lower_pad
        
        # For PyTorch, we'll use a 1x1 convolution to change dimensions instead of padding
        # as it's more efficient and common in PyTorch implementations
        shortcut = nn.Conv3d(num_fm_in, num_fm_out, kernel_size=1, stride=stride, bias=False)
        shortcut_bn = nn.BatchNorm3d(num_fm_out)
        adjusted_input = shortcut_bn(shortcut(x))
    
    # Residual connection + activation
    output = adjusted_input + layer2(layer(x))
    output = F.relu(output)
    
    return output


class Conv4d(nn.Module):
    """4D Convolution implementation using multiple 3D convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv4d, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        if isinstance(stride, int):
            stride = [stride] * 4
        if isinstance(padding, int):
            padding = [padding] * 4
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # We'll implement 4D convolution as a series of 3D convolutions
        self.conv3d = nn.Conv3d(in_channels, out_channels, 
                               kernel_size=kernel_size[1:], 
                               stride=stride[1:], 
                               padding=padding[1:], 
                               dilation=dilation, 
                               groups=groups, 
                               bias=bias)
        
    def forward(self, x):
        # x shape: [batch, channels, time, depth, height, width]
        batch_size, n_channels, n_time, depth, height, width = x.shape
        
        # Process each time slice with 3D convolution
        outputs = []
        for t in range(0, n_time, self.stride[0]):
            if t + self.kernel_size[0] > n_time:
                break
                
            # Get the temporal window
            x_t = x[:, :, t:t+self.kernel_size[0], :, :, :]
            
            # Merge time dimension with batch for 3D convolution
            x_t = x_t.transpose(1, 2).contiguous().view(-1, n_channels, depth, height, width)
            
            # Apply 3D convolution
            out_t = self.conv3d(x_t)
            
            # Reshape back
            _, out_channels, out_depth, out_height, out_width = out_t.shape
            out_t = out_t.view(batch_size, self.kernel_size[0], out_channels, out_depth, out_height, out_width)
            
            # Sum across the kernel's time dimension
            out_t = out_t.sum(dim=1)
            
            outputs.append(out_t)
            
        # Stack all time slices
        if outputs:
            return torch.stack(outputs, dim=2)
        else:
            # Handle the case when no valid outputs
            return torch.zeros(batch_size, self.conv3d.out_channels, 0, 
                              (depth - self.kernel_size[1] + 2*self.padding[1]) // self.stride[1] + 1,
                              (height - self.kernel_size[2] + 2*self.padding[2]) // self.stride[2] + 1,
                              (width - self.kernel_size[3] + 2*self.padding[3]) // self.stride[3] + 1,
                              device=x.device)


class Pool4d(nn.Module):
    """4D Pooling implementation using multiple 3D pooling operations"""
    def __init__(self, kernel_size, stride=None, padding=0, pool_type='avg'):
        super(Pool4d, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 4
        if stride is None:
            stride = kernel_size
        if isinstance(stride, int):
            stride = [stride] * 4
        if isinstance(padding, int):
            padding = [padding] * 4
            
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_type = pool_type
        
        # Create the 3D pooling layer
        if pool_type == 'avg':
            self.pool3d = nn.AvgPool3d(kernel_size=kernel_size[1:], stride=stride[1:], padding=padding[1:])
        elif pool_type == 'max':
            self.pool3d = nn.MaxPool3d(kernel_size=kernel_size[1:], stride=stride[1:], padding=padding[1:])
        else:
            raise ValueError(f"Unsupported pooling type: {pool_type}")
        
    def forward(self, x):
        # x shape: [batch, channels, time, depth, height, width]
        batch_size, n_channels, n_time, depth, height, width = x.shape
        
        # Process each time slice with 3D pooling
        outputs = []
        for t in range(0, n_time, self.stride[0]):
            end_t = min(t + self.kernel_size[0], n_time)
            if end_t - t < self.kernel_size[0]:
                # Pad if necessary
                pad_size = self.kernel_size[0] - (end_t - t)
                x_t = F.pad(x[:, :, t:end_t, :, :, :], (0, 0, 0, 0, 0, 0, 0, pad_size))
            else:
                x_t = x[:, :, t:end_t, :, :, :]
                
            # Apply time dimension pooling manually
            if self.pool_type == 'avg':
                x_t = x_t.mean(dim=2, keepdim=True)
            else:  # max pooling
                x_t, _ = x_t.max(dim=2, keepdim=True)
                
            # Reshape for 3D pooling
            x_t = x_t.squeeze(2)
            
            # Apply 3D pooling
            out_t = self.pool3d(x_t)
            
            outputs.append(out_t)
            
        # Stack all time slices
        if outputs:
            return torch.stack(outputs, dim=2)
        else:
            # Handle the case when no valid outputs
            out_depth = (depth - self.kernel_size[1] + 2*self.padding[1]) // self.stride[1] + 1
            out_height = (height - self.kernel_size[2] + 2*self.padding[2]) // self.stride[2] + 1
            out_width = (width - self.kernel_size[3] + 2*self.padding[3]) // self.stride[3] + 1
            return torch.zeros(batch_size, n_channels, 0, out_depth, out_height, out_width, device=x.device)


class Conv4dBatchNorm(nn.Module):
    """4D Convolution with BatchNorm and optional activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        super(Conv4dBatchNorm, self).__init__()
        
        self.conv4d = Conv4d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm3d(out_channels)  # Note: BatchNorm operates on the channel dimension
        self.activation = activation
        
    def forward(self, x):
        x = self.conv4d(x)
        
        # For BatchNorm, we need to handle the time dimension separately
        batch_size, n_channels, n_time, depth, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4, 5).contiguous().view(-1, n_channels, depth, height, width)
        x = self.bn(x)
        x = x.view(batch_size, n_time, n_channels, depth, height, width).permute(0, 2, 1, 3, 4, 5)
        
        if self.activation is not None:
            x = self.activation(x)
            
        return x


def resneta_block_4d(x, num_fm_out, stride, batch_size, factorize=False, kernel_size=[3,3,3,3]):
    """Defines a single resnetA block for 4D data
    Args: 
      x: block input tensor
      num_fm_out: base number of feature maps in the block
    Returns:
      output: tensor, output of the block 
    """
    # Number of input feature maps
    num_fm_in = x.size(1)
    
    # Determine if it's a reduction
    increase_dim = num_fm_out > num_fm_in
    
    # First layer
    if factorize:
        # Spatial convolution
        spatial_conv = Conv4dBatchNorm(
            in_channels=num_fm_in,
            out_channels=num_fm_out,
            kernel_size=[1, kernel_size[1], kernel_size[2], kernel_size[3]],
            stride=stride,
            activation=F.relu
        )
        
        # Temporal convolution
        temp_conv = Conv4dBatchNorm(
            in_channels=num_fm_out,
            out_channels=num_fm_out,
            kernel_size=[kernel_size[0], 1, 1, 1],
            stride=[1, 1, 1, 1],
            activation=F.relu
        )
        
        layer = temp_conv(spatial_conv(x))
    else:
        conv4d_bn = Conv4dBatchNorm(
            in_channels=num_fm_in,
            out_channels=num_fm_out,
            kernel_size=kernel_size,
            stride=stride,
            activation=F.relu
        )
        layer = conv4d_bn(x)
    
    # Second layer
    if factorize:
        # Spatial convolution
        spatial_conv2 = Conv4dBatchNorm(
            in_channels=num_fm_out,
            out_channels=num_fm_out,
            kernel_size=[1, kernel_size[1], kernel_size[2], kernel_size[3]],
            stride=[1, 1, 1, 1],
            activation=F.relu
        )
        
        # Temporal convolution (no activation)
        temp_conv2 = Conv4dBatchNorm(
            in_channels=num_fm_out,
            out_channels=num_fm_out,
            kernel_size=[kernel_size[0], 1, 1, 1],
            stride=[1, 1, 1, 1],
            activation=None
        )
        
        layer2 = temp_conv2(spatial_conv2(layer))
    else:
        conv4d_bn2 = Conv4dBatchNorm(
            in_channels=num_fm_out,
            out_channels=num_fm_out,
            kernel_size=kernel_size,
            stride=[1, 1, 1, 1],
            activation=None
        )
        layer2 = conv4d_bn2(layer)
    
    # Shortcut connection handling
    adjusted_input = x
    
    # Handle strided convolution
    if stride[1] == 2:
        pool4d = Pool4d(kernel_size=[2, 2, 2, 2], stride=[1, 2, 2, 2], pool_type='avg')
        adjusted_input = pool4d(adjusted_input)
    
    # Handle dimension increase
    if increase_dim:
        # Instead of padding, use a 1x1x1x1 convolution to match dimensions
        dim_adjust = Conv4dBatchNorm(
            in_channels=num_fm_in,
            out_channels=num_fm_out,
            kernel_size=1,
            stride=1,
            activation=None
        )
        adjusted_input = dim_adjust(adjusted_input)
    
    # Residual connection + activation
    output = F.relu(adjusted_input + layer2)
    
    return output


class BatchRNNNorm(nn.Module):
    """Batch normalization for recurrent architectures"""
    def __init__(self, num_features, eps=1e-3, momentum=0.01, affine=True, keep_time_dim=False):
        super(BatchRNNNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.keep_time_dim = keep_time_dim
        
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x, training=True):
        # x shape: [batch, time, features] or [batch, features]
        
        if self.keep_time_dim:
            # Treat each timestep independently
            # Reshape to [batch, time, 1, features] for BatchNorm2d
            orig_shape = x.shape
            if len(orig_shape) == 3:  # [batch, time, features]
                x = x.unsqueeze(2)
                if training:
                    batch_mean = x.mean(dim=(0, 1), keepdim=True)
                    batch_var = x.var(dim=(0, 1), unbiased=False, keepdim=True)
                    
                    # Update running stats
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
                else:
                    batch_mean = self.running_mean.view(1, 1, 1, -1)
                    batch_var = self.running_var.view(1, 1, 1, -1)
                
                # Normalize
                x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
                
                # Apply affine transform if requested
                if self.affine:
                    x = x * self.weight.view(1, 1, 1, -1) + self.bias.view(1, 1, 1, -1)
                
                # Reshape back
                x = x.squeeze(2)
            else:  # [batch, features]
                if training:
                    batch_mean = x.mean(dim=0, keepdim=True)
                    batch_var = x.var(dim=0, unbiased=False, keepdim=True)
                    
                    # Update running stats
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
                else:
                    batch_mean = self.running_mean.view(1, -1)
                    batch_var = self.running_var.view(1, -1)
                
                # Normalize
                x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
                
                # Apply affine transform if requested
                if self.affine:
                    x = x * self.weight.view(1, -1) + self.bias.view(1, -1)
        else:
            # Aggregate stats across time
            if len(x.shape) == 3:  # [batch, time, features]
                if training:
                    batch_mean = x.mean(dim=(0, 1), keepdim=True)
                    batch_var = x.var(dim=(0, 1), unbiased=False, keepdim=True)
                    
                    # Update running stats
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
                else:
                    batch_mean = self.running_mean.view(1, 1, -1)
                    batch_var = self.running_var.view(1, 1, -1)
                
                # Normalize
                x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
                
                # Apply affine transform if requested
                if self.affine:
                    x = x * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)
            else:  # [batch, features]
                if training:
                    batch_mean = x.mean(dim=0, keepdim=True)
                    batch_var = x.var(dim=0, unbiased=False, keepdim=True)
                    
                    # Update running stats
                    with torch.no_grad():
                        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.squeeze()
                        self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.squeeze()
                else:
                    batch_mean = self.running_mean.view(1, -1)
                    batch_var = self.running_var.view(1, -1)
                
                # Normalize
                x = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
                
                # Apply affine transform if requested
                if self.affine:
                    x = x * self.weight.view(1, -1) + self.bias.view(1, -1)
        
        return x


class BN_GRUCell(nn.Module):
    """GRU cell with Batch Normalization"""
    def __init__(self, input_size, hidden_size, activation=torch.tanh, keep_time_dim=False):
        super(BN_GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.keep_time_dim = keep_time_dim
        
        # Input projection (without bias, as BN will provide it)
        self.x2h = nn.Linear(input_size, 2 * hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        
        # Candidate projection
        self.x2c = nn.Linear(input_size, hidden_size, bias=False)
        self.rh2c = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Batch norm layers
        self.bn_x = BatchRNNNorm(2 * hidden_size, keep_time_dim=keep_time_dim)
        self.bn_x_c = BatchRNNNorm(hidden_size, keep_time_dim=keep_time_dim)
        
    def forward(self, x, h, training=True):
        # x: [batch, input_size]
        # h: [batch, hidden_size]
        
        # Gates
        x_gates = self.x2h(x)
        h_gates = self.h2h(h)
        
        # Apply batch norm to input projection
        bn_x_gates = self.bn_x(x_gates, training)
        
        # Combine and split for reset and update gates
        gates = bn_x_gates + h_gates
        r, u = torch.split(gates, self.hidden_size, dim=1)
        r = torch.sigmoid(r)
        u = torch.sigmoid(u)
        
        # Candidate state
        x_c = self.x2c(x)
        rh_c = self.rh2c(r * h)
        
        # Apply batch norm to candidate input projection
        bn_x_c = self.bn_x_c(x_c, training)
        
        # Candidate activation
        c = self.activation(bn_x_c + rh_c)
        
        # Final state update
        h_new = u * h + (1 - u) * c
        
        return h_new


class BN_LSTMCell(nn.Module):
    """LSTM cell with Batch Normalization"""
    def __init__(self, input_size, hidden_size, forget_bias=1.0, activation=torch.tanh, keep_time_dim=False):
        super(BN_LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.activation = activation
        self.keep_time_dim = keep_time_dim
        
        # Input projection (without bias, as BN will provide it)
        self.x2h = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        
        # Batch norm layer
        self.bn_x = BatchRNNNorm(4 * hidden_size, keep_time_dim=keep_time_dim)
        
    def forward(self, x, state, training=True):
        # x: [batch, input_size]
        # state: (c, h) where c, h: [batch, hidden_size]
        
        c, h = state
        
        # Gates
        x_gates = self.x2h(x)
        h_gates = self.h2h(h)
        
        # Apply batch norm to input projection
        bn_x_gates = self.bn_x(x_gates, training)
        
        # Combine and split for gates
        gates = bn_x_gates + h_gates
        j, i, f, o = torch.split(gates, self.hidden_size, dim=1)
        
        # Apply gate activations
        f = torch.sigmoid(f + self.forget_bias)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        j = self.activation(j)
        
        # Update cell state
        c_new = c * f + i * j
        
        # Update hidden state
        h_new = o * self.activation(c_new)
        
        return h_new, (c_new, h_new)


class BN_ConvGRUCell(nn.Module):
    """ConvGRU cell with Batch Normalization"""
    def __init__(self, input_shape, filters, kernel_size, activation=torch.tanh, keep_time_dim=False):
        super(BN_ConvGRUCell, self).__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.keep_time_dim = keep_time_dim
        
        # Determine padding based on kernel size
        padding = kernel_size // 2
        
        # Input projections (without bias, as BN will provide it)
        self.conv_x = nn.Conv2d(input_shape[-1], 2 * filters, kernel_size, padding=padding, bias=False)
        self.conv_h = nn.Conv2d(filters, 2 * filters, kernel_size, padding=padding, bias=True)
        
        # Candidate projections
        self.conv_x_c = nn.Conv2d(input_shape[-1], filters, kernel_size, padding=padding, bias=False)
        self.conv_rh = nn.Conv2d(filters, filters, kernel_size, padding=padding, bias=True)
        
        # Batch norm layers
        self.bn_x = BatchRNNNorm(2 * filters, keep_time_dim=keep_time_dim)
        self.bn_x_c = BatchRNNNorm(filters, keep_time_dim=keep_time_dim)
        
    def forward(self, x, h, training=True):
        # x: [batch, height, width, in_channels]
        # h: [batch, height, width, filters]
        
        # For PyTorch's Conv2d, we need channels first
        x = x.permute(0, 3, 1, 2)  # [batch, in_channels, height, width]
        h = h.permute(0, 3, 1, 2)  # [batch, filters, height, width]
        
        # Gates
        x_gates = self.conv_x(x)
        h_gates = self.conv_h(h)
        
        # Reshape for batch norm
        batch_size, channels, height, width = x_gates.shape
        x_gates = x_gates.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        h_gates = h_gates.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        
        # Apply batch norm to input projection
        bn_x_gates = self.bn_x(x_gates, training)
        
        # Reshape back
        bn_x_gates = bn_x_gates.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        h_gates = h_gates.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Combine and split for reset and update gates
        gates = bn_x_gates + h_gates
        r, u = torch.split(gates, self.filters, dim=1)
        r = torch.sigmoid(r)
        u = torch.sigmoid(u)
        
        # Candidate state
        x_c = self.conv_x_c(x)
        rh = r * h
        rh_c = self.conv_rh(rh)
        
        # Reshape for batch norm
        batch_size, channels, height, width = x_c.shape
        x_c = x_c.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        rh_c = rh_c.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        
        # Apply batch norm to candidate input projection
        bn_x_c = self.bn_x_c(x_c, training)
        
        # Reshape back
        bn_x_c = bn_x_c.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        rh_c = rh_c.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Candidate activation
        c = self.activation(bn_x_c + rh_c)
        
        # Final state update
        h_new = u * h + (1 - u) * c
        
        # Convert back to channels last format
        h_new = h_new.permute(0, 2, 3, 1)  # [batch, height, width, filters]
        
        return h_new


class Resnet4D(nn.Module):
    """4D CNN Architecture based on ResNet"""
    def __init__(self, mdl_params):
        super(Resnet4D, self).__init__()
        self.mdl_params = mdl_params
        
        # Initial layer
        self.initial_spatial = Conv4dBatchNorm(
            in_channels=1,  # Assuming single channel input
            out_channels=mdl_params['num_filters_init'],
            kernel_size=[1, mdl_params['kernel_init'][1], mdl_params['kernel_init'][2], mdl_params['kernel_init'][3]],
            stride=mdl_params['strides_init'],
            activation=F.relu
        ) if mdl_params.get('factorize', False) else None
        
        self.initial_temp = Conv4dBatchNorm(
            in_channels=mdl_params['num_filters_init'],
            out_channels=mdl_params['num_filters_init'],
            kernel_size=[mdl_params['kernel_init'][0], 1, 1, 1],
            stride=[1, 1, 1, 1],
            activation=F.relu
        ) if mdl_params.get('factorize', False) else None
        
        self.initial = Conv4dBatchNorm(
            in_channels=1,  # Assuming single channel input
            out_channels=mdl_params['num_filters_init'],
            kernel_size=mdl_params['kernel_init'],
            stride=mdl_params['strides_init'],
            activation=F.relu
        ) if not mdl_params.get('factorize', False) else None
        
        # Build resnet modules
        self.resnet_modules = nn.ModuleList()
        for i in range(len(mdl_params['ResNet_Size'])):
            blocks = nn.ModuleList()
            for j in range(mdl_params['ResNet_Size'][i]):
                if j == 0:
                    output_fm = mdl_params['ResNet_FM'][i]
                    block_stride = mdl_params['ResNet_Stride'][i]
                else:
                    block_stride = [1, 1, 1, 1]
                
                # Create a custom ResNet block
                blocks.append(lambda x, output_fm=output_fm, block_stride=block_stride: 
                             resneta_block_4d(x, output_fm, block_stride, 
                                             mdl_params['batchSize'], 
                                             mdl_params.get('factorize', False)))
            
            self.resnet_modules.append(blocks)
        
        # Fully connected layer
        self.fc = nn.Linear(mdl_params['ResNet_FM'][-1], mdl_params['numOut'])
        
    def forward(self, x, training=True):
        # x: [batch, timesteps, width, height, depth, channels]
        
        # Transform to channel first: [batch, channels, timesteps, width, height, depth]
        x = x.permute(0, 5, 1, 2, 3, 4)
        
        # Initial convolution
        if self.mdl_params.get('factorize', False):
            x = self.initial_spatial(x)
            x = self.initial_temp(x)
        else:
            x = self.initial(x)
        
        # ResNet modules
        for module in self.resnet_modules:
            for block in module:
                x = block(x)
        
        # Global Average Pooling
        x = x.mean(dim=(2, 3, 4, 5))
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class Resnet3D(nn.Module):
    """3D CNN Architecture based on ResNet"""
    def __init__(self, mdl_params):
        super(Resnet3D, self).__init__()
        self.mdl_params = mdl_params
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv3d(
                in_channels=1,  # Assuming single channel input
                out_channels=mdl_params['num_filters_init'],
                kernel_size=mdl_params['kernel_init'],
                stride=mdl_params['strides_init'],
                padding=[k//2 for k in mdl_params['kernel_init']] if isinstance(mdl_params['kernel_init'], list) else mdl_params['kernel_init']//2
            ),
            nn.BatchNorm3d(mdl_params['num_filters_init']),
            nn.ReLU(inplace=True)
        )
        
        # Build resnet modules
        self.resnet_modules = nn.ModuleList()
        for i in range(len(mdl_params['ResNet_Size'])):
            blocks = nn.ModuleList()
            for j in range(mdl_params['ResNet_Size'][i]):
                if j == 0:
                    output_fm = mdl_params['ResNet_FM'][i]
                    block_stride = mdl_params['ResNet_Stride'][i]
                else:
                    block_stride = [1, 1, 1] if '3D' in mdl_params.get('model_type', '') else [1, 1]
                
                # Create a custom ResNet block
                blocks.append(lambda x, output_fm=output_fm, block_stride=block_stride: 
                             resneta_block(x, output_fm, block_stride, 
                                          mdl_params.get('factorize', False),
                                          is_temporal='timesteps' in mdl_params))
            
            self.resnet_modules.append(blocks)
        
        # Fully connected layer
        self.fc = nn.Linear(mdl_params['ResNet_FM'][-1], mdl_params['numOut'])
        
    def forward(self, x, training=True):
        # x could be 3D, 4D, or 5D depending on input
        
        # Initial convolution
        x = self.initial(x)
        
        # ResNet modules
        for module in self.resnet_modules:
            for block in module:
                x = block(x)
        
        # Global Average Pooling - handle different input dimensions
        if x.dim() == 3:  # [batch, channels, time]
            x = x.mean(dim=2)
        elif x.dim() == 4:  # [batch, channels, height, width]
            x = x.mean(dim=(2, 3))
        elif x.dim() == 5:  # [batch, channels, time, height, width]
            x = x.mean(dim=(2, 3, 4))
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class ConvGRUCNN(nn.Module):
    """ConvGRU-CNN architecture"""
    def __init__(self, mdl_params):
        super(ConvGRUCNN, self).__init__()
        self.mdl_params = mdl_params
        
        # ConvGRU layers
        self.conv_gru_layers = nn.ModuleList()
        for i in range(mdl_params['num_convGRU_layers']):
            if mdl_params.get('use_lstm', False):
                cell = BN_ConvLSTMCell(
                    input_shape=mdl_params['input_size'],
                    filters=mdl_params['num_hidden'],
                    kernel_size=mdl_params['convgru_kernel'],
                    keep_time_dim=mdl_params['keep_time_dim']
                )
            else:
                cell = BN_ConvGRUCell(
                    input_shape=mdl_params['input_size'],
                    filters=mdl_params['num_hidden'],
                    kernel_size=mdl_params['convgru_kernel'],
                    keep_time_dim=mdl_params['keep_time_dim']
                )
            self.conv_gru_layers.append(cell)
        
        # Initial CNN convolution
        self.initial = nn.Sequential(
            nn.Conv3d(
                in_channels=mdl_params['num_hidden'],  # Output from ConvGRU
                out_channels=mdl_params['num_filters_init'],
                kernel_size=mdl_params['kernel_init'],
                stride=mdl_params['strides_init'],
                padding=[k//2 for k in mdl_params['kernel_init']] if isinstance(mdl_params['kernel_init'], list) else mdl_params['kernel_init']//2
            ),
            nn.BatchNorm3d(mdl_params['num_filters_init']),
            nn.ReLU(inplace=True)
        )
        
        # Build resnet modules for CNN part
        self.resnet_modules = nn.ModuleList()
        for i in range(len(mdl_params['ResNet_Size'])):
            blocks = nn.ModuleList()
            for j in range(mdl_params['ResNet_Size'][i]):
                if j == 0:
                    output_fm = mdl_params['ResNet_FM'][i]
                    block_stride = mdl_params['ResNet_Stride'][i]
                else:
                    if len(mdl_params['input_size']) == 3:
                        block_stride = [1, 1, 1]
                    elif len(mdl_params['input_size']) == 2:
                        block_stride = [1, 1]
                
                kernel_size = [3, 3, 3] if len(mdl_params['input_size']) == 3 else [3, 3]
                
                # Create a custom ResNet block
                blocks.append(lambda x, output_fm=output_fm, block_stride=block_stride, kernel_size=kernel_size: 
                             resneta_block(x, output_fm, block_stride, kernel_size=kernel_size))
            
            self.resnet_modules.append(blocks)
        
        # Fully connected layer
        self.fc = nn.Linear(mdl_params['ResNet_FM'][-1], mdl_params['numOut'])
        
    def forward(self, x, training=True):
        # x: [batch, timesteps, height, width, channels] or similar
        
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Process through ConvGRU/ConvLSTM layers
        h_state = None
        for t in range(time_steps):
            x_t = x[:, t]  # Get current time step
            
            for i, cell in enumerate(self.conv_gru_layers):
                if i == 0:
                    if h_state is None:
                        # Initialize hidden state for first layer
                        if isinstance(cell, BN_ConvLSTMCell):
                            # For LSTM, initialize both hidden and cell state
                            h = torch.zeros(batch_size, *self.mdl_params['input_size'][:-1], self.mdl_params['num_hidden'], 
                                           device=x.device)
                            c = torch.zeros_like(h)
                            h_state = [(c, h)]
                        else:
                            # For GRU, initialize just the hidden state
                            h = torch.zeros(batch_size, *self.mdl_params['input_size'][:-1], self.mdl_params['num_hidden'], 
                                           device=x.device)
                            h_state = [h]
                    
                    if isinstance(cell, BN_ConvLSTMCell):
                        h_t, h_state[i] = cell(x_t, h_state[i], training)
                    else:
                        h_t = cell(x_t, h_state[i], training)
                        h_state[i] = h_t
                else:
                    if len(h_state) <= i:
                        # Initialize hidden state for subsequent layers
                        if isinstance(cell, BN_ConvLSTMCell):
                            h = torch.zeros_like(h_t)
                            c = torch.zeros_like(h)
                            h_state.append((c, h))
                        else:
                            h = torch.zeros_like(h_t)
                            h_state.append(h)
                    
                    if isinstance(cell, BN_ConvLSTMCell):
                        h_t, h_state[i] = cell(h_t, h_state[i], training)
                    else:
                        h_t = cell(h_t, h_state[i], training)
                        h_state[i] = h_t
        
        # Use the last output
        x = h_t
        
        # For CNN part, convert to channels first format
        x = x.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        
        # Initial CNN layer
        x = self.initial(x)
        
        # ResNet modules
        for module in self.resnet_modules:
            for block in module:
                x = block(x)
        
        # Global Average Pooling - handle different input dimensions
        if x.dim() == 3:  # [batch, channels, dim]
            x = x.mean(dim=2)
        elif x.dim() == 4:  # [batch, channels, height, width]
            x = x.mean(dim=(2, 3))
        elif x.dim() == 5:  # [batch, channels, depth, height, width]
            x = x.mean(dim=(2, 3, 4))
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class CNNGRU(nn.Module):
    """CNN-GRU architecture"""
    def __init__(self, mdl_params):
        super(CNNGRU, self).__init__()
        self.mdl_params = mdl_params
        
        # Initial CNN convolution
        if len(mdl_params.get('input_size', [])) == 3:  # 4D input
            self.initial = nn.Sequential(
                nn.Conv3d(
                    in_channels=1,  # Assuming single channel input
                    out_channels=mdl_params['num_filters_init'],
                    kernel_size=mdl_params['kernel_init'],
                    stride=mdl_params['strides_init'],
                    padding=[k//2 for k in mdl_params['kernel_init']] if isinstance(mdl_params['kernel_init'], list) else mdl_params['kernel_init']//2
                ),
                nn.BatchNorm3d(mdl_params['num_filters_init']),
                nn.ReLU(inplace=True)
            )
        else:  # 6D input for 4D convolutions
            self.initial = Conv4dBatchNorm(
                in_channels=1,  # Assuming single channel input
                out_channels=mdl_params['num_filters_init'],
                kernel_size=mdl_params['kernel_init'],
                stride=mdl_params['strides_init'],
                activation=F.relu
            )
        
        # Build resnet modules for CNN part
        self.resnet_modules = nn.ModuleList()
        for i in range(len(mdl_params['ResNet_Size'])):
            blocks = nn.ModuleList()
            for j in range(mdl_params['ResNet_Size'][i]):
                if j == 0:
                    output_fm = mdl_params['ResNet_FM'][i]
                    block_stride = mdl_params['ResNet_Stride'][i]
                else:
                    if mdl_params.get('input_size', None) and len(mdl_params['input_size']) == 3:
                        block_stride = [1, 1, 1]
                    else:
                        block_stride = [1, 1, 1, 1]
                
                # Create a custom ResNet block based on input dimensionality
                if mdl_params.get('input_size', None) and len(mdl_params['input_size']) == 3:
                    blocks.append(lambda x, output_fm=output_fm, block_stride=block_stride, kernel_size=mdl_params['ResNet_Kernels'][i]: 
                                resneta_block(x, output_fm, block_stride, kernel_size=kernel_size, is_temporal=True))
                else:
                    blocks.append(lambda x, output_fm=output_fm, block_stride=block_stride: 
                                resneta_block_4d(x, output_fm, block_stride, mdl_params['batchSize'], False, mdl_params['ResNet_Kernels'][i]))
            
            self.resnet_modules.append(blocks)
        
        # GRU/LSTM Layers
        self.rnn_layers = nn.ModuleList()
        for i in range(mdl_params['num_GRU_layers']):
            if mdl_params.get('use_lstm', False):
                cell = BN_LSTMCell(
                    input_size=mdl_params['ResNet_FM'][-1],  # Using output from CNN
                    hidden_size=mdl_params['num_hidden'],
                    keep_time_dim=mdl_params['keep_time_dim']
                )
            else:
                cell = BN_GRUCell(
                    input_size=mdl_params['ResNet_FM'][-1],  # Using output from CNN
                    hidden_size=mdl_params['num_hidden'],
                    keep_time_dim=mdl_params['keep_time_dim']
                )
            
            # Add dropout
            self.rnn_layers.append((cell, mdl_params.get('dropout_rate', 0.5)))
        
        # Fully connected layer
        self.fc = nn.Linear(mdl_params['num_hidden'], mdl_params['numOut'])
    
    def forward(self, x, training=True):
        batch_size = x.size(0)
        time_steps = x.size(1)
        
        # Process each time step through CNN first
        cnn_features = []
        
        for t in range(time_steps):
            x_t = x[:, t]  # Get current time step
            
            # For 4D convolution, reshape accordingly
            if len(self.mdl_params.get('input_size', [])) != 3:  # 6D input
                x_t = x_t.permute(0, 4, 1, 2, 3)  # [batch, channels, depth, height, width]
            
            # Initial CNN layer
            features = self.initial(x_t)
            
            # ResNet modules
            for module in self.resnet_modules:
                for block in module:
                    features = block(features)
            
            # Global Average Pooling for CNN features if specified
            if self.mdl_params.get('use_pooled_features', False):
                if features.dim() == 5:  # [batch, channels, time, height, width]
                    features = features.mean(dim=(3, 4))
                elif features.dim() == 6:  # For 4D convolution output
                    features = features.mean(dim=(3, 4, 5))
            
            cnn_features.append(features)
        
        # Stack CNN features along time dimension
        if self.mdl_params.get('use_pooled_features', False):
            x = torch.stack(cnn_features, dim=1)  # [batch, time, channels, remaining_dims...]
        else:
            # Reshape for RNN input
            x = torch.stack(cnn_features, dim=1)
            x = x.view(batch_size, time_steps, -1)  # Flatten spatial dimensions
        
        # Process through GRU/LSTM layers
        h_state = None
        for i, (cell, dropout_rate) in enumerate(self.rnn_layers):
            outputs = []
            
            if h_state is None or len(h_state) <= i:
                # Initialize hidden state
                if isinstance(cell, BN_LSTMCell):
                    h = torch.zeros(batch_size, cell.hidden_size, device=x.device)
                    c = torch.zeros_like(h)
                    current_state = (c, h)
                else:
                    current_state = torch.zeros(batch_size, cell.hidden_size, device=x.device)
                
                if h_state is None:
                    h_state = [current_state]
                else:
                    h_state.append(current_state)
            else:
                current_state = h_state[i]
            
            # Process each time step
            for t in range(time_steps):
                x_t = x[:, t]
                
                # Apply dropout to the input (except for the first layer)
                if i > 0 and training and dropout_rate > 0:
                    x_t = F.dropout(x_t, p=dropout_rate, training=training)
                
                if isinstance(cell, BN_LSTMCell):
                    h_t, current_state = cell(x_t, current_state, training)
                else:
                    h_t = cell(x_t, current_state, training)
                    current_state = h_t
                
                # Apply dropout to the output
                if training and dropout_rate > 0:
                    h_t = F.dropout(h_t, p=dropout_rate, training=training)
                
                outputs.append(h_t)
                h_state[i] = current_state
            
            # Use output from last time step only
            x = outputs[-1]
        
        # Fully connected layer
        x = self.fc(x)
        
        return x


class BN_ConvLSTMCell(nn.Module):
    """ConvLSTM cell with Batch Normalization"""
    def __init__(self, input_shape, filters, kernel_size, forget_bias=1.0, activation=torch.tanh, keep_time_dim=False):
        super(BN_ConvLSTMCell, self).__init__()
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.forget_bias = forget_bias
        self.activation = activation
        self.keep_time_dim = keep_time_dim
        
        # Determine padding based on kernel size
        padding = kernel_size // 2
        
        # Input projections (without bias, as BN will provide it)
        self.conv_x = nn.Conv2d(input_shape[-1], 4 * filters, kernel_size, padding=padding, bias=False)
        self.conv_h = nn.Conv2d(filters, 4 * filters, kernel_size, padding=padding, bias=True)
        
        # Batch norm layer
        self.bn_x = BatchRNNNorm(4 * filters, keep_time_dim=keep_time_dim)
        
    def forward(self, x, state, training=True):
        # x: [batch, height, width, in_channels]
        # state: (c, h) where c, h: [batch, height, width, filters]
        
        c, h = state
        
        # For PyTorch's Conv2d, we need channels first
        x = x.permute(0, 3, 1, 2)  # [batch, in_channels, height, width]
        h = h.permute(0, 3, 1, 2)  # [batch, filters, height, width]
        c = c.permute(0, 3, 1, 2)  # [batch, filters, height, width]
        
        # Gates
        x_gates = self.conv_x(x)
        h_gates = self.conv_h(h)
        
        # Reshape for batch norm
        batch_size, channels, height, width = x_gates.shape
        x_gates = x_gates.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        h_gates = h_gates.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        
        # Apply batch norm to input projection
        bn_x_gates = self.bn_x(x_gates, training)
        
        # Reshape back
        bn_x_gates = bn_x_gates.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        h_gates = h_gates.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        # Combine and split for gates
        gates = bn_x_gates + h_gates
        j, i, f, o = torch.split(gates, self.filters, dim=1)
        
        # Apply gate activations
        f = torch.sigmoid(f + self.forget_bias)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        j = self.activation(j)
        
        # Update cell state
        c_new = c * f + i * j
        
        # Update hidden state
        h_new = o * self.activation(c_new)
        
        # Convert back to channels last format
        c_new = c_new.permute(0, 2, 3, 1)  # [batch, height, width, filters]
        h_new = h_new.permute(0, 2, 3, 1)  # [batch, height, width, filters]
        
        return h_new, (c_new, h_new)