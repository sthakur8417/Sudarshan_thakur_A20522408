import torch
import torch.nn as nn
from src.model.keypointspooling import LeftPool, TopPool, BottomPool,RightPool

j=0

class ConvBnRelu(nn.Module):
    """docstring for BnReluConv"""
    def __init__(self, inChannels, outChannels, kernelSize=1, stride=1, padding=0):
        super(ConvBnRelu, self).__init__()
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        print("==========conv",self.inChannels,self.outChannels)
        self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding,bias=self.inChannels)
        self.bn = nn.BatchNorm2d(self.outChannels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BnReluConv(nn.Module):
        """docstring for BnReluConv"""
        def __init__(self, inChannels, outChannels, kernelSize = 1, stride = 1, padding = 0):
                super(BnReluConv, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.kernelSize = kernelSize
                self.stride = stride
                self.padding = padding

                self.bn = nn.BatchNorm2d(self.inChannels)
                self.conv = nn.Conv2d(self.inChannels, self.outChannels, self.kernelSize, self.stride, self.padding)
                self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
                x = self.bn(x)
                x = self.relu(x)
                x = self.conv(x)
                return x


class ConvBlock(nn.Module):
        """docstring for ConvBlock"""
        def __init__(self, inChannels, outChannels):
                super(ConvBlock, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.outChannelsby2 = outChannels//2

                self.cbr1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)
                self.cbr2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
                self.cbr3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

        def forward(self, x):
                x = self.cbr1(x)
                x = self.cbr2(x)
                x = self.cbr3(x)
                return x

class SkipLayer(nn.Module):
        """docstring for SkipLayer"""
        def __init__(self, inChannels, outChannels):
                super(SkipLayer, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                if (self.inChannels == self.outChannels):
                        self.conv = None
                else:
                        self.conv = nn.Conv2d(self.inChannels, self.outChannels, 1)

        def forward(self, x):
                if self.conv is not None:
                        x = self.conv(x)
                return x

class Residual(nn.Module):
        """docstring for Residual"""
        def __init__(self, inChannels, outChannels):
                super(Residual, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.cb = ConvBlock(inChannels, outChannels)
                self.skip = SkipLayer(inChannels, outChannels)

        def forward(self, x):
                out = 0
                out = out + self.cb(x)
                out = out + self.skip(x)
                return out


class FeaEnhance(nn.Module):
        """docstring for Residual"""
        def __init__(self, inChannels, outChannels):
                super(FeaEnhance, self).__init__()
                super(FeaEnhance, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels
                self.outChannelsby2 = outChannels // 2


                self.atrousconv1 = nn.Conv2d(self.inChannels, self.outChannelsby2, 3, stride=2, dilation=2, padding=2)
                self.maxp1 = nn.MaxPool2d(2, 2)
                self.conv1 = nn.Conv2d(self.inChannels, self.outChannelsby2, 1, 1, 0)
                self.bnrc1 = BnReluConv(self.inChannels, self.outChannelsby2, 1, 1, 0)


                self.maxp2 = nn.MaxPool2d(2, 2)
                self.upsampling1 = nn.Upsample(scale_factor=2)
                self.maxp3 = nn.MaxPool2d(2, 2)
                self.upsampling2 = nn.Upsample(scale_factor=2)

                self.bnrc2 = BnReluConv(self.outChannelsby2, self.outChannelsby2, 3, 1, 1)
                self.atrousconv2 = nn.Conv2d(self.outChannelsby2, self.outChannelsby2, 3, stride=4, dilation=2,
                                             padding=2)
                self.maxp4 = nn.MaxPool2d(4, 4)


                self.upsampling4 = nn.Upsample(scale_factor=4)
                self.conv2 = nn.Conv2d(self.outChannelsby2, self.outChannels, 1, 1, 0)
                self.upsampling5 = nn.Upsample(scale_factor=4)
                self.conv3 = nn.Conv2d(self.outChannelsby2, self.outChannels, 1, 1, 0)

                self.bnrc3 = BnReluConv(self.outChannelsby2, self.outChannels, 1, 1, 0)

                self.skip = SkipLayer(inChannels, outChannels)

        def forward(self, x):



            out2 = self.conv1(out2)
            out3 = self.bnrc1(x)


            out1s2_1 = self.maxp2(out1)
            out1s2_2 = self.upsampling1(out1)

            out2s2_1 = self.maxp3(out2)
            out2s2_2 = self.upsampling2(out2)

            out3s2_1 = self.atrousconv2(out3)
            out3s2_2 = self.maxp4(out3)
            out3s2_3 = self.bnrc2(out3)

            out1s2 = out1s2_1 + out3s2_1
            out2s2 = out2s2_1 + out3s2_2
            out3s2 = out3s2_3 + out1s2_2 + out2s2_2


            out1s3 = self.upsampling4(out1s2)
            out1s3 = self.conv2(out1s3)

            out2s3 = self.upsampling5(out2s2)
            out2s3 = self.conv3(out2s3)

            out3s3 = self.bnrc3(out3s2)

            out = out1s3 + out3s3 + out2s3
            out = out + self.skip(x)
            return out



class Feaenhance_2(nn.Module):
        """docstring for Feaenhance"""
        def __init__(self, inChannels, outChannels):
                super(Feaenhance_2, self).__init__()
                self.inChannels = inChannels
                self.outChannels = outChannels



                self.bnrc1 = BnReluConv(self.inChannels, self.outChannels, 1, 1, 0)


                self.bnrc2 = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)
                self.maxp1 = nn.MaxPool2d(2, 2)
                self.maxp2 = nn.MaxPool2d(4, 4)


                self.bnrc3 = BnReluConv(self.outChannels, self.outChannels, 1, 1, 0)
                self.maxp3 = nn.MaxPool2d(2, 2)
                self.maxp4 = nn.MaxPool2d(4, 4)

                self.bnrc4 = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)
                self.conv1 = nn.Conv2d(self.outChannels, self.outChannels, 1, 1, 0)
                self.upsampling1 = nn.Upsample(scale_factor=2)
                self.maxp5 = nn.MaxPool2d(2, 2)
              


                self.bnrc5 = BnReluConv(self.outChannels, self.outChannels, 3, 1, 1)
                self.conv2 =  nn.Conv2d(self.outChannels, self.outChannels, 1, 1, 0)
                self.upsampling2= nn.Upsample(scale_factor=4)
                self.upsampling3 = nn.Upsample(scale_factor=2)

                self.upsampling4 = nn.Upsample(scale_factor=2)
                self.upsampling5 = nn.Upsample(scale_factor=4)

                self.conv3 =  nn.Conv2d(self.outChannels * 3, self.outChannels, 1, 1, 0)


        def forward(self, x):
            out1 = self.bnrc1(x)

            out2 = self.bnrc2(out1)
            out2_1 = self.maxp1(out1)
            out2_2 = self.maxp2(out1)




            out3 = self.bnrc3(out2)
            out3_1 = self.maxp3(out2)
            out3_2 = self.maxp4(out2)

            out4 = self.bnrc4(out2_1)
            out4_1 = self.upsampling1(self.conv1(out2_1))
            out4_2 = self.maxp5(self.conv1(out2_1))

            out5 = self.bnrc5(out2_2)
            out5_1 = self.upsampling2(self.conv2(out2_2))
            out5_2 = self.upsampling3(self.conv2(out2_2))

            out3 = out3 + out4_1 + out5_1
            out4 = out4 + out3_1 + out5_2
            out5 = out5 + out3_2 + out4_2

            out4 = self.upsampling4(out4)
            out5 = self.upsampling5(out5)

            out = torch.cat((out3, out4), dim=1)
            out = torch.cat((out, out5), dim=1)

            out = self.conv3(out)
            return out




class MeanFieldUpdate(nn.Module):
    """
    Meanfield updating

    """


        super(MeanFieldUpdate, self).__init__()




        self.attenmap_g = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1)
        self.attenmap_a = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3, stride=1, padding=1)


        self.norm_atten_a = nn.Sigmoid()
        self.norm_atten_g = nn.Sigmoid()


        self.message_f = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                   stride=1, padding=1)

        self.message_f_2 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                   stride=1, padding=1)

        self.update3 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,
                                   stride=1, padding=1)
                                
        self.update3_2 = nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=3,  stride=1, padding=1)





        ori_x_S = x_S
        

        ori_x_S = self.attenmap(ori_x_S)
        g_s = ori_x_S.mul(x_s)

        x_S = self.attenmap_2(x_S) 
        a_s_0 = x_S.mul(x_s)



     
        g_s = self.norm_atten_g(g_s+g_s_1)
        a_s = self.norm_atten_a(a_s_0+a_s_1)



        y_s = self.message_f(x_s)
        y_S = y_s.mul(g_s)  

        y_s_a = self.message_f_2(x_s)
        y_S_a = y_s_a.mul(a_s) 


        



        y_S_1 = self.update3(y_S)


        y_S_a = self.update3_2(y_S)
        y_s_a = y_S_a.mul(a_s)



        return y_S, y_s

class attentionCRF(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(attentionCRF, self).__init__()


        self.meanFieldUpdate1_1 = MeanFieldUpdate(inChannels, outChannels)
        self.meanFieldUpdate1_2 = MeanFieldUpdate(inChannels, outChannels)
        self.meanFieldUpdate1_3 = MeanFieldUpdate(inChannels, outChannels)
        self.meanFieldUpdate1_4 = MeanFieldUpdate(inChannels, outChannels)



    def forward(self, lpool, tpool, rpool, bpool, p):


        y_S, y_s_1 = self.meanFieldUpdate1_1(lpool, p)
        y_S, y_s_2 = self.meanFieldUpdate1_2(tpool, y_S)
        y_S, y_s_3 = self.meanFieldUpdate1_3(rpool, y_S)
        y_S, y_s_4 = self.meanFieldUpdate1_4(bpool, y_S)

        for i  in range(1):
            y_S, y_s_1 = self.meanFieldUpdate1_1(y_s_1, y_S)
            y_S, y_s_2 = self.meanFieldUpdate1_2(y_s_2, y_S)
            y_S, y_s_3 = self.meanFieldUpdate1_3(y_s_3, y_S)
            y_S, y_s_3 = self.meanFieldUpdate1_4(y_s_4, y_S)


        return y_S


class keypointspool(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(keypointspool, self).__init__()

        self.p2_conv1 = ConvBnRelu(inChannels, outChannels, 1)

        self.p4_conv1 = ConvBnRelu(inChannels, outChannels, 1)


        self.p_conv1 = nn.Conv2d(outChannels, outChannels, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(outChannels)

        self.conv1 = nn.Conv2d(outChannels, outChannels, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(outChannels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = ConvBnRelu(outChannels, outChannels, 1)

        self.pool1 = LeftPool()
        self.pool2 = TopPool()
        self.pool3 = RightPool()
        self.pool4 = BottomPool()

        self.atCRF = attentionCRF(inChannels, outChannels)

    def forward(self, x):


        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)


        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)


        p3_conv1 = self.p3_conv1(x)
        pool3 = self.pool3(p1_conv1)


        p4_conv1 = self.p4_conv1(x)
        pool4 = self.pool4(p2_conv1)


        confusionfea = self.atCRF(pool1, pool2, pool3, pool4)
        p_conv1 = self.p_conv1(confusionfea)
        p_bn1   = self.p_bn1(p_conv1)





        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2



class SSMixer(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(SSMixer, self).__init__()

        self.p2_conv1 = ConvBnRelu(inChannels, outChannels, 1)

        self.p4_conv1 = ConvBnRelu(inChannels, outChannels, 1)
        self.p5_conv1 = ConvBnRelu(inChannels, outChannels, 1)

        self.conv2 = ConvBnRelu(inChannels, outChannels, 1)

        self.pool1 = LeftPool()
        self.pool2 = TopPool()
        self.pool3 = RightPool()
        self.pool4 = BottomPool()

        self.atCRF = attentionCRF(inChannels, outChannels)

    def forward(self, x):
        global j

        p1_conv1 = self.p1_conv1(x)
        pool1    = self.pool1(p1_conv1)


        p2_conv1 = self.p2_conv1(x)
        pool2    = self.pool2(p2_conv1)


        p3_conv1 = self.p3_conv1(x)
        pool3 = self.pool3(p1_conv1)


        p4_conv1 = self.p4_conv1(x)
        pool4 = self.pool4(p2_conv1)

        p5 = self.p5_conv1(x)






        confusionfea = self.atCRF(pool1, pool2, pool3, pool4,p5)

        conv2 = self.conv2(confusionfea)
        return conv2





class myUpsample(nn.Module):
     def __init__(self):
         super(myUpsample, self).__init__()
         pass
     def forward(self, x):
         return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2)*2, x.size(3)*2)
