import torch.nn as nn
import torch

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)


def check_valid_activation(choice):
    if choice not in ['relu', 'lrelu', 'prelu']:
        raise ValueError(f"'{choice}' is not a valid activation function. Choose among ['relu', 'lrelu', 'prelu'].\n")


def upconv(in_channels, out_channels, mode='transpose'):
    # stride=2 implies upsampling by a factor of 2
    get_up_mode = nn.ModuleDict([
        ['bilinear', nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), conv1x1(in_channels, out_channels))],
        ['transpose', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)]
    ])

    return get_up_mode[mode]


def get_activation(choice):
    activation_functions = nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['lrelu', nn.LeakyReLU(inplace=True)],
        ['prelu', nn.PReLU()]
        ])
    return activation_functions[choice]


def conv_block(in_channels, out_channels, activation='relu', do_BN=True, *args, **kwargs):
    """
    Partial encoder block consisting of a 3×3 convolutional layer with stride 1, followed by batch normalization
    (optional) and a non-linear activation function.
    """

    if do_BN:
        return nn.Sequential(
            conv3x3(in_channels, out_channels, bias=False, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )
    else:
        return nn.Sequential(
            conv3x3(in_channels, out_channels, bias=True, *args, **kwargs),
            get_activation(activation)
        )


def conv_up_block(in_channels, out_channels, activation='relu', do_BN=True, up_mode='transpose', *args, **kwargs):
    """
    Decoder block consisting of an up-convolutional layer, followed by a 3×3 convolutional layer with stride 1,
    batch normalization (optional), and a non-linear activation function.
    """

    if do_BN:
        return nn.Sequential(
            upconv(in_channels, in_channels, up_mode),
            nn.Sequential(
                conv3x3(in_channels, out_channels, bias=False, *args, **kwargs),
                nn.BatchNorm2d(out_channels),
                get_activation(activation))
            )
    else:
        return nn.Sequential(
            upconv(in_channels, in_channels, up_mode),
            nn.Sequential(
                conv3x3(in_channels, out_channels, bias=True, *args, **kwargs),
                get_activation(activation))
            )


def bottleneck(in_channels, out_channels, activation='relu', do_BN=True, *args, **kwargs):
    """
    Bottleneck block.
    """

    if do_BN:
        return nn.Sequential(
            conv3x3(in_channels, out_channels, bias=False, *args, **kwargs),
            nn.BatchNorm2d(out_channels),
            get_activation(activation)
        )
    else:
        return nn.Sequential(
            conv3x3(in_channels, out_channels, bias=True, *args, **kwargs),
            get_activation(activation)
        )


class SkipConnection(nn.Module):
    def __init__(self):
        super(SkipConnection, self).__init__()

    def forward(self, x_skip, x_up):
        return x_skip + x_up

class UNet(nn.Module): #raw
    def __init__(self, n_input_channels=1, start_kernel=64, max_filter_depth=512, depth=8,
                 act_fn_encoder='relu', act_fn_decoder='relu', act_fn_bottleneck='relu', up_mode='transpose',
                 do_BN=True, bias_conv_layer=False, outer_skip=True, outer_skip_BN=False):
        """
        UNet network architecture.
        :param n_input_channels:    int, number of input channels
        :param start_kernel:        int, number of filters of the first convolutional layer in the encoder
        :param max_filter_depth:    int, maximum filter depth
        :param depth:               int, number of downsampling and upsampling layers (i.e., number of blocks in the
                                    encoder and decoder)
        :param act_fn_encoder:      str, activation function used in the encoder
        :param act_fn_decoder:      str, activation function used in the decoder
        :param act_fn_bottleneck:   str, activation function used in the bottleneck
        :param up_mode:             str, upsampling mode
        :param do_BN:               boolean, True to perform batch normalization after every convolutional layer,
                                    False otherwise
        :param bias_conv_layer:     boolean, True to activate the learnable bias of the convolutional layers,
                                    False otherwise
        :param outer_skip:          boolean, True to activate the long residual skip connection that adds the
                                    initial DSM to the output of the last decoder layer, False otherwise
        :param outer_skip_BN:       boolean, True to add batch normalization to the long residual skip connection,
                                    False otherwise
        """
        super(UNet, self).__init__()
        check_valid_activation(act_fn_encoder)
        check_valid_activation(act_fn_decoder)
        check_valid_activation(act_fn_bottleneck)
        if up_mode not in ['transpose', 'bilinear']:
            raise ValueError(f"'{up_mode}' is not a valid mode for upsampling. Choose among ['transpose', 'bilinear'] " "to specify 'up_mode'.\n")
        self.n_input_channels = n_input_channels
        self.start_kernel = start_kernel
        self.depth = depth
        self.act_fn_encoder = act_fn_encoder
        self.act_fn_decoder = act_fn_decoder
        self.act_fn_bottleneck = act_fn_bottleneck
        self.up_mode = up_mode
        self.max_filter_depth = max_filter_depth
        self.do_BN = do_BN
        self.bias_conv_layer = bias_conv_layer
        self.do_outer_skip = outer_skip
        self.do_outer_skip_BN = outer_skip_BN
        self.filter_depths = [self.start_kernel * (2 ** i) for i in range(self.depth)]
        # Restrict the maximum filter depth to a predefined value
        self.filter_depths = [self.max_filter_depth if i > self.max_filter_depth else i for i in self.filter_depths]
        # Set up the encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            conv_block(self.n_input_channels, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        # Set up the bottleneck
        self.bottleneck = bottleneck(self.filter_depths[-1], self.filter_depths[-1], activation=self.act_fn_bottleneck, do_BN=self.do_BN)
        # Set up the decoder
        self.decoder = nn.ModuleList()
        self.filter_depths_up = list(reversed(self.filter_depths))
        for in_channel, out_channel in zip(self.filter_depths_up[:-1], self.filter_depths_up[1:]):
            self.decoder.append(conv_up_block(in_channel, out_channel, activation=self.act_fn_decoder, up_mode=self.up_mode, do_BN=self.do_BN))
        self.decoder.append(upconv(self.filter_depths_up[-1], self.filter_depths_up[-1], up_mode))
        # Set up the final layer of the decoder
        self.last_layer = conv3x3(self.start_kernel, 1, bias=self.bias_conv_layer)
        # Skip connection
        self.skipconnect = SkipConnection()
        # Batch normalization added to the long residual skip connection
        if self.do_outer_skip:
            self.layer_outer_skip = nn.ModuleList()
            if self.do_outer_skip_BN:
                self.layer_outer_skip.append(nn.BatchNorm2d(1))
            self.layer_outer_skip.append(SkipConnection())

    def forward(self, x):
        skip_connections = []
        out = x
        # Encoder (save intermediate outputs for skip connections)
        for index, layer in enumerate(self.encoder):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)
            out_before_pool = layer_conv(out)
            skip_connections.append(out_before_pool)
            out = layer_pool(out_before_pool)
        # Bottleneck
        out = self.bottleneck(out)
        # Decoder + skip connections
        index_max = len(self.decoder) - 1
        for index, layer in enumerate(self.decoder):
            if index <= index_max - 1:
                layer_upconv = layer[0]  # upconv layer
                layer_conv = layer[1::]  # all other layers (conv, batchnorm, activation)

                out_temp = layer_upconv(out)
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
                out = layer_conv(out)
            else:
                out_temp = layer(out)   # upconv of last layer
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
        # Last layer of the decoder
        out = self.last_layer(out)
        # Add long residual skip connection
        if self.do_outer_skip:
            if self.layer_outer_skip.__len__() == 2:
                # pipe input through a batch normalization layer before adding it to the output of the last
                # decoder layer
                bn = self.layer_outer_skip[0]
                x_0 = x[:, 0, :, :]       # use channel 0 only
                x_0 = x_0.unsqueeze(1)
                x = bn(x_0)
            # add (batchnorm) input to the output of the last decoder layer
            add = self.layer_outer_skip[-1]
            x_0 = x[:, 0, :, :]
            x_0 = x_0.unsqueeze(1)
            out = add(x_0, out)  # use channel 0 only
        return out

def image_warp(x, disp, disp2):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    disp: [B, 1, H, W] disp
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    if x.is_cuda:
        xx = xx.float().cuda()
        yy = yy.float().cuda()
    xx_warp = torch.autograd.Variable(xx) - disp
    yy_warp = torch.autograd.Variable(yy) - disp # wxq
    vgrid = torch.cat((xx_warp, yy_warp), 1) # wxq
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid) 
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    mask[mask<0.999] = 0
    mask[mask>0] = 1
    return output*mask

class UNet_Img(nn.Module): #raw
    def __init__(self, n_input_channels=1, start_kernel=64, max_filter_depth=512, depth=8,
                 act_fn_encoder='relu', act_fn_decoder='relu', act_fn_bottleneck='relu', up_mode='transpose',
                 do_BN=True, bias_conv_layer=False, outer_skip=True, outer_skip_BN=False):
        """
        UNet network architecture.
        :param n_input_channels:    int, number of input channels
        :param start_kernel:        int, number of filters of the first convolutional layer in the encoder
        :param max_filter_depth:    int, maximum filter depth
        :param depth:               int, number of downsampling and upsampling layers (i.e., number of blocks in the
                                    encoder and decoder)
        :param act_fn_encoder:      str, activation function used in the encoder
        :param act_fn_decoder:      str, activation function used in the decoder
        :param act_fn_bottleneck:   str, activation function used in the bottleneck
        :param up_mode:             str, upsampling mode
        :param do_BN:               boolean, True to perform batch normalization after every convolutional layer,
                                    False otherwise
        :param bias_conv_layer:     boolean, True to activate the learnable bias of the convolutional layers,
                                    False otherwise
        :param outer_skip:          boolean, True to activate the long residual skip connection that adds the
                                    initial DSM to the output of the last decoder layer, False otherwise
        :param outer_skip_BN:       boolean, True to add batch normalization to the long residual skip connection,
                                    False otherwise
        """
        super(UNet_Img, self).__init__()
        check_valid_activation(act_fn_encoder)
        check_valid_activation(act_fn_decoder)
        check_valid_activation(act_fn_bottleneck)
        if up_mode not in ['transpose', 'bilinear']:
            raise ValueError(f"'{up_mode}' is not a valid mode for upsampling. Choose among ['transpose', 'bilinear'] " "to specify 'up_mode'.\n")
        self.n_input_channels = n_input_channels
        self.start_kernel = start_kernel
        self.depth = depth
        self.act_fn_encoder = act_fn_encoder
        self.act_fn_decoder = act_fn_decoder
        self.act_fn_bottleneck = act_fn_bottleneck
        self.up_mode = up_mode
        self.max_filter_depth = max_filter_depth
        self.do_BN = do_BN
        self.bias_conv_layer = bias_conv_layer
        self.do_outer_skip = outer_skip
        self.do_outer_skip_BN = outer_skip_BN
        self.filter_depths = [self.start_kernel * (2 ** i) for i in range(self.depth)]
        # Restrict the maximum filter depth to a predefined value
        self.filter_depths = [self.max_filter_depth if i > self.max_filter_depth else i for i in self.filter_depths]
        # Set up the encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            conv_block(self.n_input_channels, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        # Set up the bottleneck
        self.bottleneck = bottleneck(self.filter_depths[-1], self.filter_depths[-1], activation=self.act_fn_bottleneck, do_BN=self.do_BN)
        # Set up the decoder
        self.decoder = nn.ModuleList()
        self.filter_depths_up = list(reversed(self.filter_depths))
        for in_channel, out_channel in zip(self.filter_depths_up[:-1], self.filter_depths_up[1:]):
            self.decoder.append(conv_up_block(in_channel, out_channel, activation=self.act_fn_decoder, up_mode=self.up_mode, do_BN=self.do_BN))
        self.decoder.append(upconv(self.filter_depths_up[-1], self.filter_depths_up[-1], up_mode))
        # Set up the final layer of the decoder
        self.last_layer = conv3x3(self.start_kernel, 1, bias=self.bias_conv_layer)
        # Skip connection
        self.skipconnect = SkipConnection()
        # Batch normalization added to the long residual skip connection
        if self.do_outer_skip:
            self.layer_outer_skip = nn.ModuleList()
            if self.do_outer_skip_BN:
                self.layer_outer_skip.append(nn.BatchNorm2d(1))
            self.layer_outer_skip.append(SkipConnection())
        self.dsm_res = nn.Sequential(
            conv3x3(1, 2, bias=False),
            get_activation('prelu')
        )
    def forward(self, x):
        skip_connections = []
        out = x
        # Encoder (save intermediate outputs for skip connections)
        for index, layer in enumerate(self.encoder):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)
            out_before_pool = layer_conv(out)
            skip_connections.append(out_before_pool)
            out = layer_pool(out_before_pool)
        # Bottleneck
        out = self.bottleneck(out)
        # Decoder + skip connections
        index_max = len(self.decoder) - 1
        for index, layer in enumerate(self.decoder):
            if index <= index_max - 1:
                layer_upconv = layer[0]  # upconv layer
                layer_conv = layer[1::]  # all other layers (conv, batchnorm, activation)
                out_temp = layer_upconv(out)
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
                out = layer_conv(out)
            else:
                out_temp = layer(out)   # upconv of last layer
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
        # Last layer of the decoder
        out = self.last_layer(out)
        # Add long residual skip connection
        if self.do_outer_skip:
            if self.layer_outer_skip.__len__() == 2:
                # pipe input through a batch normalization layer before adding it to the output of the last
                # decoder layer
                bn = self.layer_outer_skip[0]
                x_0 = x[:, 0, :, :]       # use channel 0 only
                x_0 = x_0.unsqueeze(1)
                x = bn(x_0)
            # add (batchnorm) input to the output of the last decoder layer
            add = self.layer_outer_skip[-1]
            x_0 = x[:, 0, :, :]
            x_0 = x_0.unsqueeze(1)

            left = x[:, 1, :, :].unsqueeze(1)
            right = x[:, 2, :, :].unsqueeze(1)
            dsm_res = self.dsm_res(out)
            dsm_res1 = dsm_res[:, 0, :, :].unsqueeze(1)
            dsm_res2 = dsm_res[:, 1, :, :].unsqueeze(1)
            left_res = image_warp(left,dsm_res1,dsm_res2)
            right_res = image_warp(right,-dsm_res1,-dsm_res2)
            LR_res = torch.concat([left_res,right_res],dim=1)

            out = add(x_0, out)  # use channel 0 only
        return out, LR_res

class UNet_MonoDSM(nn.Module): #UNET+单RGB重建DSM
    def __init__(self, n_input_channels=1, start_kernel=64, max_filter_depth=512, depth=8,
                 act_fn_encoder='relu', act_fn_decoder='relu', act_fn_bottleneck='relu', up_mode='transpose',
                 do_BN=True, bias_conv_layer=False, outer_skip=True, outer_skip_BN=False):

        super(UNet_MonoDSM, self).__init__()
        check_valid_activation(act_fn_encoder)
        check_valid_activation(act_fn_decoder)
        check_valid_activation(act_fn_bottleneck)
        if up_mode not in ['transpose', 'bilinear']:
            raise ValueError(f"'{up_mode}' is not a valid mode for upsampling. Choose among ['transpose', 'bilinear'] "
                             "to specify 'up_mode'.\n")
        self.n_input_channels = n_input_channels
        self.start_kernel = start_kernel
        self.depth = depth
        self.act_fn_encoder = act_fn_encoder
        self.act_fn_decoder = act_fn_decoder
        self.act_fn_bottleneck = act_fn_bottleneck
        self.up_mode = up_mode
        self.max_filter_depth = max_filter_depth
        self.do_BN = do_BN
        self.bias_conv_layer = bias_conv_layer
        self.do_outer_skip = outer_skip
        self.do_outer_skip_BN = outer_skip_BN
        self.filter_depths = [self.start_kernel * (2 ** i) for i in range(self.depth)]

        # Restrict the maximum filter depth to a predefined value
        self.filter_depths = [self.max_filter_depth if i > self.max_filter_depth else i for i in self.filter_depths]
        # Set up the encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            conv_block(self.n_input_channels, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))
        # Set up the bottleneck
        self.bottleneck = bottleneck(self.filter_depths[-1], self.filter_depths[-1], activation=self.act_fn_bottleneck,
                                     do_BN=self.do_BN)
        # Set up the decoder
        self.decoder = nn.ModuleList()
        self.filter_depths_up = list(reversed(self.filter_depths))
        for in_channel, out_channel in zip(self.filter_depths_up[:-1], self.filter_depths_up[1:]):
            self.decoder.append(conv_up_block(in_channel, out_channel, activation=self.act_fn_decoder,
                                              up_mode=self.up_mode, do_BN=self.do_BN))
        self.decoder.append(upconv(self.filter_depths_up[-1], self.filter_depths_up[-1], up_mode))
        # Set up the final layer of the decoder
        self.last_layer = conv3x3(self.start_kernel, 1, bias=self.bias_conv_layer)
        # Skip connection
        self.skipconnect = SkipConnection()
        # Batch normalization added to the long residual skip connection
        if self.do_outer_skip:
            self.layer_outer_skip = nn.ModuleList()
            if self.do_outer_skip_BN:
                self.layer_outer_skip.append(nn.BatchNorm2d(1))
            self.layer_outer_skip.append(SkipConnection())

    def forward(self, x):
        skip_connections = []
        lr = x[:, 0, :, :].unsqueeze(1)  #low-res dsm provide only basis
        out = x[:, 1:3, :, :].unsqueeze(1) #rgb--->dsm
        for index, layer in enumerate(self.encoder):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)
            out_before_pool = layer_conv(out)
            skip_connections.append(out_before_pool)
            out = layer_pool(out_before_pool)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder + skip connections
        index_max = len(self.decoder) - 1
        for index, layer in enumerate(self.decoder):
            if index <= index_max - 1:
                layer_upconv = layer[0]  # upconv layer
                layer_conv = layer[1::]  # all other layers (conv, batchnorm, activation)

                out_temp = layer_upconv(out)
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
                out = layer_conv(out)
            else:
                out_temp = layer(out)   # upconv of last layer
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
        # Last layer of the decoder
        out = self.last_layer(out)
        # Add long residual skip connection
        if self.do_outer_skip:
            if self.layer_outer_skip.__len__() == 2:
                # pipe input through a batch normalization layer before adding it to the output of the last
                # decoder layer
                bn = self.layer_outer_skip[0]
                lr = bn(lr)
            # add (batchnorm) input to the output of the last decoder layer
            add = self.layer_outer_skip[-1]
            out = add(out, lr)
        return out

'''
class UNet(nn.Module): # 1012 原始网络+一个注意力模块，让网络更加注意变化的区域；注意力模块先用CBAM凑合一下
    def __init__(self, n_input_channels=1, start_kernel=64, max_filter_depth=512, depth=8,
                 act_fn_encoder='relu', act_fn_decoder='relu', act_fn_bottleneck='relu', up_mode='transpose',
                 do_BN=True, bias_conv_layer=False, outer_skip=True, outer_skip_BN=False):
        """
        UNet network architecture.

        :param n_input_channels:    int, number of input channels
        :param start_kernel:        int, number of filters of the first convolutional layer in the encoder
        :param max_filter_depth:    int, maximum filter depth
        :param depth:               int, number of downsampling and upsampling layers (i.e., number of blocks in the
                                    encoder and decoder)
        :param act_fn_encoder:      str, activation function used in the encoder
        :param act_fn_decoder:      str, activation function used in the decoder
        :param act_fn_bottleneck:   str, activation function used in the bottleneck
        :param up_mode:             str, upsampling mode
        :param do_BN:               boolean, True to perform batch normalization after every convolutional layer,
                                    False otherwise
        :param bias_conv_layer:     boolean, True to activate the learnable bias of the convolutional layers,
                                    False otherwise
        :param outer_skip:          boolean, True to activate the long residual skip connection that adds the
                                    initial DSM to the output of the last decoder layer, False otherwise
        :param outer_skip_BN:       boolean, True to add batch normalization to the long residual skip connection,
                                    False otherwise
        """

        super(UNet, self).__init__()

        check_valid_activation(act_fn_encoder)
        check_valid_activation(act_fn_decoder)
        check_valid_activation(act_fn_bottleneck)

        if up_mode not in ['transpose', 'bilinear']:
            raise ValueError(f"'{up_mode}' is not a valid mode for upsampling. Choose among ['transpose', 'bilinear'] "
                             "to specify 'up_mode'.\n")

        self.n_input_channels = n_input_channels
        self.start_kernel = start_kernel
        self.depth = depth
        self.act_fn_encoder = act_fn_encoder
        self.act_fn_decoder = act_fn_decoder
        self.act_fn_bottleneck = act_fn_bottleneck
        self.up_mode = up_mode
        self.max_filter_depth = max_filter_depth
        self.do_BN = do_BN
        self.bias_conv_layer = bias_conv_layer
        self.do_outer_skip = outer_skip
        self.do_outer_skip_BN = outer_skip_BN
        self.filter_depths = [self.start_kernel * (2 ** i) for i in range(self.depth)]

        # Restrict the maximum filter depth to a predefined value
        self.filter_depths = [self.max_filter_depth if i > self.max_filter_depth else i for i in self.filter_depths]

        # Set up the encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            conv_block(self.n_input_channels, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        # Set up the bottleneck
        self.bottleneck = bottleneck(self.filter_depths[-1], self.filter_depths[-1], activation=self.act_fn_bottleneck,
                                     do_BN=self.do_BN)

        # Set up the decoder
        self.decoder = nn.ModuleList()
        self.filter_depths_up = list(reversed(self.filter_depths))

        for in_channel, out_channel in zip(self.filter_depths_up[:-1], self.filter_depths_up[1:]):
            self.decoder.append(conv_up_block(in_channel, out_channel, activation=self.act_fn_decoder,
                                              up_mode=self.up_mode, do_BN=self.do_BN))
        self.decoder.append(upconv(self.filter_depths_up[-1], self.filter_depths_up[-1], up_mode))

        # Set up the final layer of the decoder
        self.last_layer = conv3x3(self.start_kernel, 1, bias=self.bias_conv_layer)

        # Skip connection
        self.skipconnect = SkipConnection()

        # Batch normalization added to the long residual skip connection
        if self.do_outer_skip:
            self.layer_outer_skip = nn.ModuleList()
            if self.do_outer_skip_BN:
                self.layer_outer_skip.append(nn.BatchNorm2d(1))
            self.layer_outer_skip.append(SkipConnection())



        #通道注意力机制
        reduction=16
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp256=nn.Sequential(
            nn.Linear(in_features=256,out_features=256//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=256//reduction,out_features=256,bias=False)
        )
        self.mlp128=nn.Sequential(
            nn.Linear(in_features=128,out_features=128//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128//reduction,out_features=128,bias=False)
        )
        self.mlp64=nn.Sequential(
            nn.Linear(in_features=64,out_features=64//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=64//reduction,out_features=64,bias=False)
        )
        self.mlp512=nn.Sequential(
            nn.Linear(in_features=512,out_features=512//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=512//reduction,out_features=512,bias=False)
        )

        # self.mlp16=nn.Sequential(
        #     nn.Linear(in_features=16,out_features=16//reduction,bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=16//reduction,out_features=16,bias=False)
        # )
        # self.mlp8=nn.Sequential(
        #     nn.Linear(in_features=8,out_features=1,bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=1,out_features=4,bias=False)
        # )
        #声明卷积核为 3 或 7
        #assert kernel_size in (3, 7)#, 'kernel size must be 3 or 7'
        kernel_size = 7
        #进行相应的same padding填充
        padding = 3 if kernel_size == 7 else 1

        #self.conv64 = nn.Conv2d(64, 1, kernel_size, padding=padding, bias=False)
        #self.conv128 = nn.Conv2d(128, 1, kernel_size, padding=padding, bias=False)
        #self.conv32 = nn.Conv2d(32, 1, kernel_size, padding=padding, bias=False)
        #self.conv16 = nn.Conv2d(16, 1, kernel_size, padding=padding, bias=False)
        #self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()



    def forward(self, x):
        skip_connections = []
        out = x

        # Encoder (save intermediate outputs for skip connections)
        for index, layer in enumerate(self.encoder):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)
            out_before_pool = layer_conv(out)

            #通道注意力机制
            maxout = self.max_pool(out_before_pool)
            avgout=self.avg_pool(out_before_pool)
           # print(f'\nmaxout.size: {maxout.size()}\n')
            #print(f'avgout.size: {avgout.size()}\n')
            if maxout.size(1) == 128:
                maxout = self.mlp128(maxout.view(maxout.size(0),-1)) #7x7卷积填充为3，输入通道为2，输出通道为1
                avgout = self.mlp128(avgout.view(avgout.size(0),-1))
            else:
                if maxout.size(1) == 64: 
                    maxout = self.mlp64(maxout.view(maxout.size(0),-1)) #7x7卷积填充为3，输入通道为2，输出通道为1
                    avgout = self.mlp64(avgout.view(avgout.size(0),-1))
                else:
                    if maxout.size(1) == 512: 
                        maxout = self.mlp512(maxout.view(maxout.size(0),-1)) #7x7卷积填充为3，输入通道为2，输出通道为1
                        avgout = self.mlp512(avgout.view(avgout.size(0),-1))
                    else:
                        if maxout.size(1) == 16: 
                            maxout = self.mlp16(maxout.view(maxout.size(0),-1)) #7x7卷积填充为3，输入通道为2，输出通道为1
                            avgout = self.mlp16(avgout.view(avgout.size(0),-1))
                        else: 
                            if maxout.size(1) == 256: 
                                maxout = self.mlp256(maxout.view(maxout.size(0),-1)) #7x7卷积填充为3，输入通道为2，输出通道为1
                                avgout = self.mlp256(avgout.view(avgout.size(0),-1))
                            else:
                                maxout = self.mlp8(maxout.view(maxout.size(0),-1))
                                avgout = self.mlp8(avgout.view(avgout.size(0),-1))

            #maxout=self.mlp(maxout.view(maxout.size(0),-1))
            #avgout=self.avg_pool(out_before_pool)
           # avgout=self.mlp(avgout.view(avgout.size(0),-1))
            channel_out=self.sigmoid(maxout+avgout)
           # print(f'\n channel_out.size: {channel_out.size()}\n')
            channel_out=channel_out.view(out_before_pool.size(0),out_before_pool.size(1),1,1)
            channel_out=channel_out*out_before_pool
          #  print(f'channel_out2.size: {channel_out.size()}\n')

            #空间注意力机制
            avg_out = torch.mean(channel_out, dim=1, keepdim=True)  #平均池化
            max_out, _ = torch.max(channel_out, dim=1, keepdim=True) #最大池化
            #拼接操作
            avg_max_out = torch.cat([avg_out, max_out], dim=1)
           # print(f'\navg_max_out.size: {avg_max_out.size()}\n')
            # if avg_max_out.size(1) == 128:
            #     avg_max_out_2 = self.conv128(avg_max_out) #7x7卷积填充为3，输入通道为2，输出通道为1
            # else:
            #     if avg_max_out.size(1) == 64: avg_max_out_2 = self.conv64(avg_max_out) #7x7卷积填充为3，输入通道为2，输出通道为1
            #     else:
            #         if avg_max_out.size(1) == 32: avg_max_out_2 = self.conv32(avg_max_out) #7x7卷积填充为3，输入通道为2，输出通道为1
            #         else:
            #             if avg_max_out.size(1) == 16: avg_max_out_2 = self.conv16(avg_max_out) #7x7卷积填充为3，输入通道为2，输出通道为1
            #             else: 
            #                 avg_max_out_2 = self.conv1(avg_max_out)
                        
            avg_max_out_2 = self.conv1(avg_max_out) #7x7卷积填充为3，输入通道为2，输出通道为1
            spatial_attention = self.sigmoid(avg_max_out_2)
           # print(f'\nout_before_pool.size: {out_before_pool.size()}\n')
           # print(f'spatial_attention.size: {spatial_attention.size()}\n')
            out_before_pool = out_before_pool * spatial_attention

            skip_connections.append(out_before_pool)
            out = layer_pool(out_before_pool)

        # Bottleneck
        out = self.bottleneck(out)

        # Decoder + skip connections
        index_max = len(self.decoder) - 1
        for index, layer in enumerate(self.decoder):
            if index <= index_max - 1:
                layer_upconv = layer[0]  # upconv layer
                layer_conv = layer[1::]  # all other layers (conv, batchnorm, activation)

                out_temp = layer_upconv(out)
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
                out = layer_conv(out)
            else:
                out_temp = layer(out)   # upconv of last layer
                out = self.skipconnect(skip_connections[-1 - index], out_temp)

        # Last layer of the decoder
        out = self.last_layer(out)

        # Add long residual skip connection
        if self.do_outer_skip:
            if self.layer_outer_skip.__len__() == 2:
                # pipe input through a batch normalization layer before adding it to the output of the last
                # decoder layer
                bn = self.layer_outer_skip[0]
                x_0 = x[:, 0, :, :]       # use channel 0 only
                x_0 = x_0.unsqueeze(1)
                x = bn(x_0)

            # add (batchnorm) input to the output of the last decoder layer
            add = self.layer_outer_skip[-1]
            x_0 = x[:, 0, :, :]
            x_0 = x_0.unsqueeze(1)

            out = add(x_0, out)  # use channel 0 only

        return out
'''

'''
class UNet(nn.Module): 
    # unet里面 左右图分别叠加dsm输入encoder，然后卷积输出通道降为原来的一半，然后池化，在concat，跳跃连接等等 1028
    def __init__(self, n_input_channels=1, start_kernel=64, max_filter_depth=512, depth=8,
                 act_fn_encoder='relu', act_fn_decoder='relu', act_fn_bottleneck='relu', up_mode='transpose',
                 do_BN=True, bias_conv_layer=False, outer_skip=True, outer_skip_BN=False):
        """
        UNet network architecture.

        :param n_input_channels:    int, number of input channels
        :param start_kernel:        int, number of filters of the first convolutional layer in the encoder
        :param max_filter_depth:    int, maximum filter depth
        :param depth:               int, number of downsampling and upsampling layers (i.e., number of blocks in the
                                    encoder and decoder)
        :param act_fn_encoder:      str, activation function used in the encoder
        :param act_fn_decoder:      str, activation function used in the decoder
        :param act_fn_bottleneck:   str, activation function used in the bottleneck
        :param up_mode:             str, upsampling mode
        :param do_BN:               boolean, True to perform batch normalization after every convolutional layer,
                                    False otherwise
        :param bias_conv_layer:     boolean, True to activate the learnable bias of the convolutional layers,
                                    False otherwise
        :param outer_skip:          boolean, True to activate the long residual skip connection that adds the
                                    initial DSM to the output of the last decoder layer, False otherwise
        :param outer_skip_BN:       boolean, True to add batch normalization to the long residual skip connection,
                                    False otherwise
        """

        super(UNet, self).__init__()

        check_valid_activation(act_fn_encoder)
        check_valid_activation(act_fn_decoder)
        check_valid_activation(act_fn_bottleneck)

        if up_mode not in ['transpose', 'bilinear']:
            raise ValueError(f"'{up_mode}' is not a valid mode for upsampling. Choose among ['transpose', 'bilinear'] "
                             "to specify 'up_mode'.\n")

        self.n_input_channels = n_input_channels
        self.start_kernel = start_kernel
        self.depth = depth
        self.act_fn_encoder = act_fn_encoder
        self.act_fn_decoder = act_fn_decoder
        self.act_fn_bottleneck = act_fn_bottleneck
        self.up_mode = up_mode
        self.max_filter_depth = max_filter_depth
        self.do_BN = do_BN
        self.bias_conv_layer = bias_conv_layer
        self.do_outer_skip = outer_skip
        self.do_outer_skip_BN = outer_skip_BN
        self.filter_depths = [self.start_kernel * (2 ** i) for i in range(self.depth)]

        # Restrict the maximum filter depth to a predefined value
        self.filter_depths = [self.max_filter_depth if i > self.max_filter_depth else i for i in self.filter_depths]

        # Set up the encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            conv_block(self.n_input_channels, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        # Set up the bottleneck
        self.bottleneck = bottleneck(self.filter_depths[-1], self.filter_depths[-1], activation=self.act_fn_bottleneck,
                                     do_BN=self.do_BN)

        # Set up the decoder
        self.decoder = nn.ModuleList()
        self.filter_depths_up = list(reversed(self.filter_depths))

        for in_channel, out_channel in zip(self.filter_depths_up[:-1], self.filter_depths_up[1:]):
            self.decoder.append(conv_up_block(in_channel, out_channel, activation=self.act_fn_decoder,
                                              up_mode=self.up_mode, do_BN=self.do_BN))
        self.decoder.append(upconv(self.filter_depths_up[-1], self.filter_depths_up[-1], up_mode))

        # Set up the final layer of the decoder
        self.last_layer = conv3x3(self.start_kernel, 1, bias=self.bias_conv_layer)

        # Skip connection
        self.skipconnect = SkipConnection()

        # Batch normalization added to the long residual skip connection
        if self.do_outer_skip:
            self.layer_outer_skip = nn.ModuleList()
            if self.do_outer_skip_BN:
                self.layer_outer_skip.append(nn.BatchNorm2d(1))
            self.layer_outer_skip.append(SkipConnection())

        self.conv256 =  nn.Conv2d(256, 128, kernel_size = 3, padding=1, bias=True)
        self.conv128 = nn.Conv2d(128, 64, kernel_size = 3, padding=1, bias=True)
        self.conv512 = nn.Conv2d(512, 256, kernel_size = 3, padding=1, bias=True)
        self.conv1024 = nn.Conv2d(1024, 512, kernel_size = 3, padding=1, bias=True)

       # self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, padding=1, bias=False) # mask
        #self.sigmoid = nn.Sigmoid() # mask

    def forward(self, x):
        skip_connections = []
        out = x[:, 0, :, :].unsqueeze(1)
        left = x[:, 1, :, :].unsqueeze(1)
        right = x[:, 2, :, :].unsqueeze(1) 
        outleft = torch.concat([out,left],dim=1)
        outright = torch.concat([out,right],dim=1)

        # Encoder (save intermediate outputs for skip connections)
        for index, layer in enumerate(self.encoder):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)

            out_before_pool_left = layer_conv(outleft)
            #print(f'\nout_before_pool_left.size: {out_before_pool_left.size()}\n')
            out_before_pool_right = layer_conv(outright)
            #print(f'\nout_before_pool_right.size: {out_before_pool_right.size()}\n')
            out_before_pool = torch.cat([out_before_pool_left, out_before_pool_right], dim=1)
           # print(f'\nout_before_pool.size: {out_before_pool.size()}\n')
            if out_before_pool.size(1) == 128:
                out_before_pool = self.conv128(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
            else:
                if out_before_pool.size(1) == 256: out_before_pool = self.conv256(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
                else:
                    if out_before_pool.size(1) == 512: out_before_pool = self.conv512(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
                    else:
                        if out_before_pool.size(1) == 1024: out_before_pool = self.conv1024(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
                        else: 
                            print(f'\nout_before_pool.size: {out_before_pool.size()}\n')

            #print(f'\nout_before_pool.size: {out_before_pool.size()}\n')
            #空间注意力机制
            #diff_spatial = torch.abs(out_before_pool_left - out_before_pool_right)# mask
            #avg_out = torch.mean(diff_spatial, dim=1, keepdim=True)  #平均池化# mask
            #max_out, _ = torch.max(diff_spatial, dim=1, keepdim=True) #最大池化# mask
            #拼接操作
            #avg_max_out = torch.cat([avg_out, max_out], dim=1)# mask
            #print(f'\navg_max_out.size: {avg_max_out.size()}\n')

            #avg_max_out_2 = self.conv1(avg_max_out) #7x7卷积填充为3，输入通道为2，输出通道为1# mask
           # spatial_attention = self.sigmoid(avg_max_out_2) # mask
            #print(f'\navg_max_out_2.size: {avg_max_out_2.size()}\n')
            #print(f'\nspatial_attention.size: {spatial_attention.size()}\n')
            #skip_connections.append(out_before_pool * spatial_attention) # 加注意力mask
            skip_connections.append(out_before_pool)
            outleft = layer_pool(out_before_pool_left)
            outright = layer_pool(out_before_pool_right)
           

        # Bottleneck
        out = layer_pool(out_before_pool) 
        out = self.bottleneck(out)

        # Decoder + skip connections
        index_max = len(self.decoder) - 1
        for index, layer in enumerate(self.decoder):
            if index <= index_max - 1:
                layer_upconv = layer[0]  # upconv layer
                layer_conv = layer[1::]  # all other layers (conv, batchnorm, activation)

                out_temp = layer_upconv(out)
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
                out = layer_conv(out)
            else:
                out_temp = layer(out)   # upconv of last layer
                out = self.skipconnect(skip_connections[-1 - index], out_temp)

        # Last layer of the decoder
        out = self.last_layer(out)

        # Add long residual skip connection
        if self.do_outer_skip:
            if self.layer_outer_skip.__len__() == 2:
                # pipe input through a batch normalization layer before adding it to the output of the last
                # decoder layer
                bn = self.layer_outer_skip[0]
                x_0 = x[:, 0, :, :]       # use channel 0 only
                x_0 = x_0.unsqueeze(1)
                x = bn(x_0)

            # add (batchnorm) input to the output of the last decoder layer
            add = self.layer_outer_skip[-1]
            x_0 = x[:, 0, :, :]
            x_0 = x_0.unsqueeze(1)

            out = add(x_0, out)  # use channel 0 only

        return out
        '''
'''
class UNet(nn.Module):   ### 1226
    # unet里面 左右图叠加输入encoder1,dsm输入encoder2，然后池化，再concat，跳跃连接等等 (测试不同的组合方式，dsm单独过encoder  left+right encoder)

    def __init__(self, n_input_channels=1, start_kernel=64, max_filter_depth=512, depth=8,
                 act_fn_encoder='relu', act_fn_decoder='relu', act_fn_bottleneck='relu', up_mode='transpose',
                 do_BN=True, bias_conv_layer=False, outer_skip=True, outer_skip_BN=False):
        """
        UNet network architecture.

        :param n_input_channels:    int, number of input channels
        :param start_kernel:        int, number of filters of the first convolutional layer in the encoder
        :param max_filter_depth:    int, maximum filter depth
        :param depth:               int, number of downsampling and upsampling layers (i.e., number of blocks in the
                                    encoder and decoder)
        :param act_fn_encoder:      str, activation function used in the encoder
        :param act_fn_decoder:      str, activation function used in the decoder
        :param act_fn_bottleneck:   str, activation function used in the bottleneck
        :param up_mode:             str, upsampling mode
        :param do_BN:               boolean, True to perform batch normalization after every convolutional layer,
                                    False otherwise
        :param bias_conv_layer:     boolean, True to activate the learnable bias of the convolutional layers,
                                    False otherwise
        :param outer_skip:          boolean, True to activate the long residual skip connection that adds the
                                    initial DSM to the output of the last decoder layer, False otherwise
        :param outer_skip_BN:       boolean, True to add batch normalization to the long residual skip connection,
                                    False otherwise
        """

        super(UNet, self).__init__()

        check_valid_activation(act_fn_encoder)
        check_valid_activation(act_fn_decoder)
        check_valid_activation(act_fn_bottleneck)

        if up_mode not in ['transpose', 'bilinear']:
            raise ValueError(f"'{up_mode}' is not a valid mode for upsampling. Choose among ['transpose', 'bilinear'] "
                             "to specify 'up_mode'.\n")

        self.n_input_channels = n_input_channels
        self.start_kernel = start_kernel
        self.depth = depth
        self.act_fn_encoder = act_fn_encoder
        self.act_fn_decoder = act_fn_decoder
        self.act_fn_bottleneck = act_fn_bottleneck
        self.up_mode = up_mode
        self.max_filter_depth = max_filter_depth
        self.do_BN = do_BN
        self.bias_conv_layer = bias_conv_layer
        self.do_outer_skip = outer_skip
        self.do_outer_skip_BN = outer_skip_BN
        self.filter_depths = [self.start_kernel * (2 ** i) for i in range(self.depth)]

        # Restrict the maximum filter depth to a predefined value
        self.filter_depths = [self.max_filter_depth if i > self.max_filter_depth else i for i in self.filter_depths]

        # Set up the encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Sequential(
            conv_block(self.n_input_channels, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        self.encoder2 = nn.ModuleList()
        self.encoder2.append(nn.Sequential(
            conv_block(1, self.start_kernel, activation=self.act_fn_encoder, do_BN=self.do_BN),
            nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        for in_channel, out_channel in zip(self.filter_depths, self.filter_depths[1:]):
            self.encoder2.append(nn.Sequential(
                conv_block(in_channel, out_channel, activation=self.act_fn_encoder, do_BN=self.do_BN),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        # Set up the bottleneck
        self.bottleneck = bottleneck(self.filter_depths[-1], self.filter_depths[-1], activation=self.act_fn_bottleneck,
                                     do_BN=self.do_BN)

        # Set up the decoder
        self.decoder = nn.ModuleList()
        self.filter_depths_up = list(reversed(self.filter_depths))

        for in_channel, out_channel in zip(self.filter_depths_up[:-1], self.filter_depths_up[1:]):
            self.decoder.append(conv_up_block(in_channel, out_channel, activation=self.act_fn_decoder,
                                              up_mode=self.up_mode, do_BN=self.do_BN))
        self.decoder.append(upconv(self.filter_depths_up[-1], self.filter_depths_up[-1], up_mode))

        # Set up the final layer of the decoder
        self.last_layer = conv3x3(self.start_kernel, 1, bias=self.bias_conv_layer)

        # Skip connection
        self.skipconnect = SkipConnection()

        # Batch normalization added to the long residual skip connection
        if self.do_outer_skip:
            self.layer_outer_skip = nn.ModuleList()
            if self.do_outer_skip_BN:
                self.layer_outer_skip.append(nn.BatchNorm2d(1))
            self.layer_outer_skip.append(SkipConnection())

        self.conv256 =  nn.Conv2d(256, 128, kernel_size = 3, padding=1, bias=True)
        self.conv128 = nn.Conv2d(128, 64, kernel_size = 3, padding=1, bias=True)
        self.conv512 = nn.Conv2d(512, 256, kernel_size = 3, padding=1, bias=True)
        self.conv1024 = nn.Conv2d(1024, 512, kernel_size = 3, padding=1, bias=True)

       # self.conv1 = nn.Conv2d(2, 1, kernel_size = 3, padding=1, bias=False) # mask
        #self.sigmoid = nn.Sigmoid() # mask

    def forward(self, x):
        skip_connections = []
        out = x[:, 0, :, :].unsqueeze(1)
        left = x[:, 1, :, :].unsqueeze(1)
        right = x[:, 2, :, :].unsqueeze(1) 
        left_right = torch.concat([left,right],dim=1)
        dsm0 = x[:, 0, :, :].unsqueeze(1)

        im_encode = []
        dsm_encode = []

        # Encoder (save intermediate outputs for skip connections)
        for index, layer in enumerate(self.encoder):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)
            out_before_pool_im = layer_conv(left_right)
            #print(f'\nout_before_pool_im.size: {out_before_pool_im.size()}\n')
            left_right = layer_pool(out_before_pool_im)
            im_encode.append(out_before_pool_im)

        for index, layer in enumerate(self.encoder2):
            layer_conv = layer[:-1]  # all layers before the pooling layer (at depth index)
            layer_pool = layer[-1]   # pooling layer (at depth index)
            out_before_pool_dsm = layer_conv(dsm0)
            #print(f'\nout_before_pool_dsm.size: {out_before_pool_dsm.size()}\n')
            dsm0 = layer_pool(out_before_pool_dsm)
            dsm_encode.append(out_before_pool_dsm)

        for index, layer in enumerate(self.encoder):
            out_before_pool_im = im_encode[index]
            out_before_pool_dsm = dsm_encode[index]
            out_before_pool = torch.cat([out_before_pool_im, out_before_pool_dsm], dim=1)
            if out_before_pool.size(1) == 128:
                out_before_pool = self.conv128(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
            else:
                if out_before_pool.size(1) == 256: out_before_pool = self.conv256(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
                else:
                    if out_before_pool.size(1) == 512: out_before_pool = self.conv512(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
                    else:
                        if out_before_pool.size(1) == 1024: out_before_pool = self.conv1024(out_before_pool) #7x7卷积填充为3，输入通道为2，输出通道为1
                        else: 
                            print(f'\nout_before_pool.size: {out_before_pool.size()}\n')
            skip_connections.append(out_before_pool)

        # Bottleneck
        out = layer_pool(out_before_pool) 
        out = self.bottleneck(out)

        # Decoder + skip connections
        index_max = len(self.decoder) - 1
        for index, layer in enumerate(self.decoder):
            if index <= index_max - 1:
                layer_upconv = layer[0]  # upconv layer
                layer_conv = layer[1::]  # all other layers (conv, batchnorm, activation)

                out_temp = layer_upconv(out)
                out = self.skipconnect(skip_connections[-1 - index], out_temp)
                out = layer_conv(out)
            else:
                out_temp = layer(out)   # upconv of last layer
                out = self.skipconnect(skip_connections[-1 - index], out_temp)

        # Last layer of the decoder
        out = self.last_layer(out)

        # Add long residual skip connection
        if self.do_outer_skip:
            if self.layer_outer_skip.__len__() == 2:
                # pipe input through a batch normalization layer before adding it to the output of the last
                # decoder layer
                bn = self.layer_outer_skip[0]
                x_0 = x[:, 0, :, :]       # use channel 0 only
                x_0 = x_0.unsqueeze(1)
                x = bn(x_0)

            # add (batchnorm) input to the output of the last decoder layer
            add = self.layer_outer_skip[-1]
            x_0 = x[:, 0, :, :]
            x_0 = x_0.unsqueeze(1)

            out = add(x_0, out)  # use channel 0 only

        return out
'''