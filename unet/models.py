# Libraries
import torch
from torch import nn

class Conv_Block(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Conv_Block, self).__init__()
        
        self.conv_1 = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding = 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 1)
    
    def forward(self, x):
        conv_1 = self.conv_1(x)
        batch_norm_1 = self.batch_norm(conv_1)
        act_1 = self.act(batch_norm_1)
        
        conv_2 = self.conv_2(act_1)
        batch_norm_2 = self.batch_norm(conv_2)
        act_2 = self.act(batch_norm_2)
        
        return act_2 

class Encoder(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Encoder, self).__init__()
        self.max_pool = nn.MaxPool2d((2, 2))
        self.conv_op = Conv_Block(input_channels, out_channels)
    
    def forward(self, x):
        enc = self.conv_op(x)
        max_pool = self.max_pool(enc)
        
        return enc, max_pool

class Decoder(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(Decoder, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_channels, out_channels, kernel_size=2, stride=2)
        self.conv_op = Conv_Block(2 * out_channels, out_channels)
        
    def forward(self, x, y):
        upsampled = self.upsample(x)
        connect_skip = torch.cat([upsampled, y], dim=1)
        out = self.conv_op(connect_skip)
        
        return out

class UNetM(nn.Module):
    def __init__(self, num_filter, num_classes):
        super(UNetM, self).__init__()
        self.e1 = Encoder(3, num_filter)
        self.e2 = Encoder(num_filter, num_filter * 2)
        self.e3 = Encoder(num_filter * 2, num_filter * 4)
        self.e4 = Encoder(num_filter * 4, num_filter * 8)
        
        self.b = Conv_Block(num_filter * 8, num_filter * 16)
        
        self.d1 = Decoder(num_filter * 16, num_filter * 8)
        self.d2 = Decoder(num_filter * 8, num_filter * 4)
        self.d3 = Decoder(num_filter * 4, num_filter * 2)
        self.d4 = Decoder(num_filter * 2, num_filter)
        
        self.output = nn.Conv2d(num_filter, num_classes, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ### Encoder ###

        skip_1, encoder_1 = self.e1(x)
        skip_2, encoder_2 = self.e2(encoder_1)
        skip_3, encoder_3 = self.e3(encoder_2)
        skip_4, encoder_4 = self.e4(encoder_3)
        
        ### BottleNeck ###
        conv_block = self.b(encoder_4)
        
        ### Decoder ###
        decoder_1 = self.d1(conv_block, skip_4)
        decoder_2 = self.d2(decoder_1, skip_3)
        decoder_3 = self.d3(decoder_2, skip_2)
        decoder_4 = self.d4(decoder_3, skip_1)
        
        out = self.output(decoder_4)
        out = self.sigmoid(out)
        
        return out
    
class UNet(nn.Module):
    def __init__(self, num_filters, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes

        self.down_conv_11 = self.conv_block(in_channels=3,
                                            out_channels=num_filters)
        self.down_conv_12 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.down_conv_21 = self.conv_block(in_channels=num_filters,
                                            out_channels=num_filters * 2)
        self.down_conv_22 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.down_conv_31 = self.conv_block(in_channels=num_filters * 2,
                                            out_channels=num_filters * 4)
        self.down_conv_32 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        self.down_conv_41 = self.conv_block(in_channels=num_filters * 4,
                                            out_channels=num_filters * 8)
        self.down_conv_42 = nn.MaxPool2d(kernel_size=2,
                                         stride=2)
        
        self.middle = self.conv_block(in_channels=num_filters * 8, out_channels=num_filters * 16)
        
        self.up_conv_11 = nn.ConvTranspose2d(in_channels=num_filters * 16, out_channels=num_filters * 8,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_12 = self.conv_block(in_channels=num_filters * 16,
                                          out_channels=num_filters * 8)
        self.up_conv_21 = nn.ConvTranspose2d(in_channels=num_filters * 8, out_channels=num_filters*4,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_22 = self.conv_block(in_channels=num_filters*8,
                                          out_channels=num_filters*4)
        self.up_conv_31 = nn.ConvTranspose2d(in_channels=num_filters*4, out_channels=num_filters*2,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_32 = self.conv_block(in_channels=num_filters*4,
                                          out_channels=num_filters * 2)
        self.up_conv_41 = nn.ConvTranspose2d(in_channels=num_filters *2, out_channels=num_filters,
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1)
        self.up_conv_42 = self.conv_block(in_channels=num_filters*2,
                                          out_channels=num_filters)
        
        self.output = nn.Conv2d(in_channels=num_filters, out_channels=num_classes,
                                kernel_size=3, stride=1,
                                padding=1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    @staticmethod
    def conv_block(in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels))
        return block
    
    @staticmethod
    def crop_tensor(target_tensor, tensor):
        target_size = target_tensor.size()[2]
        tensor_size = tensor.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2

        return tensor[:,:, delta:tensor_size-delta, delta:tensor_size-delta]


    def forward(self, X):
        x1 = self.down_conv_11(X) 
        x2 = self.down_conv_12(x1)
        x3 = self.down_conv_21(x2)
        x4 = self.down_conv_22(x3)
        x5 = self.down_conv_31(x4)
        x6 = self.down_conv_32(x5)
        x7 = self.down_conv_41(x6)
        x8 = self.down_conv_42(x7)
        
        middle_out = self.middle(x8) 

        x = self.up_conv_11(middle_out) 
        y = self.crop_tensor(x, x7)
        x = self.up_conv_12(torch.cat((x, y), dim=1)) 
        
        x = self.up_conv_21(x) 
        y = self.crop_tensor(x, x5)
        x = self.up_conv_22(torch.cat((x, y), dim=1)) 
        
        x = self.up_conv_31(x)
        y = self.crop_tensor(x, x3)
        x = self.up_conv_32(torch.cat((x, y), dim=1)) 
        
        x = self.up_conv_41(x)
        y = self.crop_tensor(x, x1)
        x = self.up_conv_42(torch.cat((x, y), dim=1)) 
        
        output = self.output(x) 
        # output = self.softmax(output)
        output = self.sigmoid(output)

        return output