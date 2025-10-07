import torch
import torch.nn as nn
import torch.nn.functional as F


#DownSampling Block of the Unet to decrease the sample to the latent space
class DownSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, dilatation=1, kernel_size=15, stride=1, padding =7):

        super(DownSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride = stride, padding = padding, dilation=dilatation),
                      nn.BatchNorm1d(channel_out),
                      nn.LeakyReLU(negative_slope=0.1)
        )

    def forward(self, ipt):
        return self.main(ipt)

class UpSamplingLayer(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding =2):
        super(UpSamplingLayer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride = stride, padding = padding),
                      nn.BatchNorm1d(channel_out),
                        nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    )
        

    def forward(self, ipt):
        return self.main(ipt)

class Model(nn.Module):
    def __init__(self, n_layers=12, channels_interval=24):
        super(Model, self).__init__()

        self.n_layers = n_layers
        self.channels_interval = channels_interval
        encoder_in_channels_list = [1] + [i * channels_interval for i in range(1, n_layers)]
        encoder_out_channels_list = [i * channels_interval for i in range(1, n_layers+1)]

        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(DownSamplingLayer(encoder_in_channels_list[i],
                                                   encoder_out_channels_list[i]))
            
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval,
                        kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )        
        # Create a list for the decoder input and output channels for the U geometric
        decoder_in_channels_list = [(2*i+1)*self.channels_interval for i in range(1, n_layers)] + [2*n_layers*channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]

        self.decoder = nn.ModuleList() # Use a list to store the decoder layers
        for i in range(self.n_layers):
            self.decoder.append(UpSamplingLayer(decoder_in_channels_list[i],
                                                decoder_out_channels_list[i]))
        self.out1 = nn.Sequential(
            nn.Conv1d(1+self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

        self.out2 = nn.Sequential(
            nn.Conv1d(1+self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, input):
        
        tmp = []
        o = input


        # Up Sampling
        for i in range(self.n_layers):
            o = self.encoder[i](o)

            tmp.append(o)
            o = o[:, :, ::2]


        o = self.middle(o)


        # Down Sampling
        for i in range(self.n_layers):
            o = F.interpolate(o, scale_factor=2, mode='linear', align_corners=True)

            skip_tensor = tmp[self.n_layers - i - 1]

            


            
            o = torch.cat((o, skip_tensor), dim=1)

            o = self.decoder[i](o)

            
        o = torch.cat([o, input], dim=1)

        o1 = self.out1(o)
        o2 = self.out2(o)
        
        return o1, o2
