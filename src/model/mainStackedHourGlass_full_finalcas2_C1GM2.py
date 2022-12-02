import torch
import torch.nn as nn
import src.model.gm_twogate_ag_cov1 as M


class myUpsample(nn.Module):
    def __init__(self):
        super(myUpsample, self).__init__()
        pass

    def forward(self, x):
        return x[:, :, :, None, :, None].expand(-1, -1, -1, 2, -1, 2).reshape(x.size(0), x.size(1), x.size(2) * 2,

                                                                              x.size(3) * 2)


class Hourglass(nn.Module):
    """docstring for Hourglass"""

    def __init__(self, nChannels=128, numReductions=4, nModules=1, poolKernel=(2, 2), poolStride=(2, 2),
                 upSampleKernel=2):
        super(Hourglass, self).__init__()
        self.numReductions = numReductions
        self.nModules = nModules
        self.nChannels = nChannels
        self.poolKernel = poolKernel
        self.poolStride = poolStride

        self.upSampleKernel = upSampleKernel

        """
        For the skip connection, a residual module (or sequence of residuaql modules)
        """

        _skip = []
        for _ in range(self.nModules):
            _skip.append(M.Residual(self.nChannels, self.nChannels))

        self.skip = nn.Sequential(*_skip)

        """
        First pooling to go to smaller dimension then pass input through
        Residual Module or sequence of Modules then  and subsequent cases:
            either pass through Hourglass of numReductions-1
            or pass through M.Residual Module or sequence of Modules
        """

        self.mp = nn.MaxPool2d(self.poolKernel, self.poolStride)

        _afterpool = []
        for _ in range(self.nModules):
            _afterpool.append(M.Residual(self.nChannels, self.nChannels))

        self.afterpool = nn.Sequential(*_afterpool)

        if (numReductions > 1):
            self.hg = Hourglass(self.nChannels, self.numReductions - 1, self.nModules, self.poolKernel, self.poolStride)
        else:
            _num1res = []
            for _ in range(self.nModules):
                _num1res.append(M.Residual(self.nChannels, self.nChannels))



        """
        Now another M.Residual Module or sequence of M.Residual Modules
        """

        _lowres = []
        for _ in range(self.nModules):
            _lowres.append(M.Residual(self.nChannels, self.nChannels))

        self.lowres = nn.Sequential(*_lowres)

        """
        As per Newell's paper upsamping recommended
        """


    def forward(self, x):
        out1 = x
        out1 = self.skip(out1)
        out2 = x


        out2 = self.mp(out2)
        out2 = self.afterpool(out2)
        if self.numReductions > 1:
            out2 = self.hg(out2)
        else:
            out2 = self.num1res(out2)
        out2 = self.lowres(out2)
        out2 = self.up(out2)

        return out2 + out1


class Gmscenet(nn.Module):
    """docstring for StackedHourGlass"""

    def __init__(self, nChannels, nStack, nModules, numReductions, nJoints):
        super(Gmscenet, self).__init__()
        self.nChannels = nChannels
        self.nStack = nStack
        self.nModules = nModules
        self.numReductions = numReductions
        self.nJoints = nJoints






        self.res1 = M.Residual(64, 128)
        self.mp = nn.MaxPool2d(2, 2)
        self.res2 = M.Residual(128, 128)
        self.res3 = M.Residual(128, self.nChannels)

        _hourglass, _Residual, _lin1, _chantojoints, _lin2, _jointstochan, _pool, _lpool, _bpool, _rbool = [], [], [], [], [], [], [], [], [], []
        _lin0, _chantojoints0 = [], []
        _lin0_1, _chantojoints0_1 = [], []
        _lin2_1, _jointstochan_1 = [], []
        _deconv, _maxpool1, _maxpool2 = [], [], []
        _Residual_deconv = []
        _lin1_1 = []
        _downsample = []

        _upsample1 = []
        _upsample2 = []

        for i in range(self.nStack):
            _hourglass.append(Hourglass(self.nChannels, self.numReductions, self.nModules))

            if(i < self.nStack - 1):


                    _ResidualModules = []
                    for _ in range(self.nModules):
                        _ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
                    _ResidualModules = nn.Sequential(*_ResidualModules)
                    _Residual.append(_ResidualModules)

                    _Residual_deconv.append(M.Residual(self.nChannels, self.nChannels))

                    _lin0.append(M.ConvBnRelu(self.nChannels, self.nChannels))
                    _chantojoints0.append(nn.Conv2d(self.nChannels, self.nJoints, 1))

                    _lin0_1.append(M.ConvBnRelu(self.nChannels, self.nChannels))
                    _chantojoints0_1.append(nn.Conv2d(self.nChannels, self.nJoints, 1))

                    _lin2.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
                    _jointstochan.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

                    _lin2_1.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
                    _jointstochan_1.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

                    _deconv.append(nn.ConvTranspose2d(2*self.nChannels, self.nChannels, kernel_size=3, stride=2, padding=1,
                                                      output_padding=1))
                  
                    _maxpool1.append(nn.MaxPool2d(2, 2))
                    _maxpool2.append(nn.MaxPool2d(2, 2))







            
            if (i == self.nStack - 1):

                    _ResidualModules = []
                    for _ in range(self.nModules):
                        _ResidualModules.append(M.Residual(self.nChannels, self.nChannels))
                    _ResidualModules = nn.Sequential(*_ResidualModules)
                    _Residual.append(_ResidualModules)

                    _Residual_deconv.append(M.Residual(self.nChannels, self.nChannels))

                    _lin0.append(M.ConvBnRelu(self.nChannels, self.nChannels))
                    _chantojoints0.append(nn.Conv2d(self.nChannels, self.nJoints, 1))

                    _lin0_1.append(M.ConvBnRelu(self.nChannels, self.nChannels))
                    _chantojoints0_1.append(nn.Conv2d(self.nChannels, self.nJoints, 1))

                    _lin2.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
                    _jointstochan.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

                    _lin2_1.append(nn.Conv2d(self.nChannels, self.nChannels, 1))
                    _jointstochan_1.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

                    _deconv.append(nn.ConvTranspose2d(2*self.nChannels, self.nChannels, kernel_size=3, stride=2, padding=1,
                                           output_padding=1))

                    _maxpool1.append(nn.MaxPool2d(2, 2))
                    _maxpool2.append(nn.MaxPool2d(2, 2))



            if (i == self.nStack - 1):
                _lin1.append(nn.Conv2d(self.nChannels * self.nStack, self.nChannels, 1))
                _lin1_1.append(nn.Conv2d(self.nJoints, self.nChannels, 1))

                for _ in range(self.nJoints):
                    _pool.append(M.SSMixer(self.nChannels, self.nChannels))
                    _chantojoints.append(nn.Conv2d(self.nChannels, 1, 1))

        self.hourglass = nn.ModuleList(_hourglass)
        self.Residual = nn.ModuleList(_Residual)
        self.Residual_deconv = nn.ModuleList(_Residual_deconv)

        self.lin0 = nn.ModuleList(_lin0)
        self.chantojoints0 = nn.ModuleList(_chantojoints0)

        self.lin0_1 = nn.ModuleList(_lin0_1)
        self.chantojoints0_1 = nn.ModuleList(_chantojoints0_1)
       
        self.deconv = nn.ModuleList(_deconv)
        self.maxpool1 = nn.ModuleList(_maxpool1)
        self.maxpool2 = nn.ModuleList(_maxpool2)







        self.lin1 = nn.ModuleList(_lin1)
        self.lin1_1 = nn.ModuleList(_lin1_1)

        self.pool = nn.ModuleList(_pool)

        self.chantojoints = nn.ModuleList(_chantojoints)

        self.lin2 = nn.ModuleList(_lin2)
        self.jointstochan = nn.ModuleList(_jointstochan)

        self.lin2_1 = nn.ModuleList(_lin2_1)
        self.jointstochan_1 = nn.ModuleList(_jointstochan_1)

    def forward(self, x):
        x = self.start(x)
        x = self.res1(x)
        x = self.mp(x)
        x = self.res2(x)
        x = self.res3(x)



        out = []

        featuremap = []

        for i in range(self.nStack):
            feamap = self.hourglass[i](x)
            featuremap.append(feamap)




            if (i < self.nStack - 1):
                for k in range(1):
                    x1 = self.Residual[i + k](feamap)
                    x1 = self.lin0[i + k](x1)
                    out.append(self.chantojoints0[ i + k](x1))

                    x1 = self.lin2[i](x1)





                    x1 = self.Residual_deconv[i + k](x1)
                    x1 = self.lin0_1[i](x1)


                    x1 = self.maxpool1[i + k](x1)
                    if(k == 0):
                        x1 = self.lin2_1[i + k](x1)
                        x = x + x1 + self.jointstochan_1[i + k](self.maxpool2[i + k](out2[i + k]))


            if (i == self.nStack - 1):
                poolinput = featuremap[0]
                for k in range(i):
                    poolinput = torch.cat((poolinput, featuremap[k + 1]), dim=1)



                for j in range(self.nJoints):
                    p[j] = self.pool[j](poolinput)
                    kout[j] = self.chantojoints[j](p[j])





                pointsoutput = kout[0]
                for m in range(self.nJoints - 1):
                    pointsoutput = torch.cat((kout[m + 1], pointsoutput), dim=1)





                afterpool = self.lin1_1[0](out[i])
                afterpool = afterpool + poolinput

                for k in range(2):
                    x1 = self.Residual[i + k](afterpool)
                    x1 = self.lin0[i + k](x1)


                    x1 = self.lin2[i + k](x1)





                    x1 = self.Residual_deconv[ i + k](x1)
                    x1 = self.lin0_1[i + k](x1)


                    x1 = self.maxpool1[i + k](x1)
                    if (k == 0):
                        x1 = self.lin2_1[i + k](x1)
                        afterpool = afterpool + x1 + self.jointstochan_1[i + k](self.maxpool2[i + k](out2[ i + k]))








        return out, out2
