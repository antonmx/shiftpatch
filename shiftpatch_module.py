
import IPython

import sys
import os
import random
import time
import gc
import dataclasses
from dataclasses import dataclass, field
from enum import Enum

import math
import statistics
from cv2 import norm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torchvision
from torch import optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread, imsave
import h5py
import tifffile
import tqdm


def initIfNew(var, val=None) :
    if var in locals() :
        return locals()[var]
    if var in globals() :
        return globals()[var]
    return val


@dataclass
class TCfgClass:
    exec : int
    latentDim: int
    batchSize: int
    labelSmoothFac: float
    learningRateD: float
    learningRateG: float
    device: torch.device = torch.device('cpu')
    batchSplit : int = 1
    nofEpochs: int = 0
    historyHDF : str = field(repr = True, init = False)
    logDir : str = field(repr = True, init = False)
    def __post_init__(self):
        if self.device == torch.device('cpu')  :
            self.device = torch.device(f"cuda:{self.exec}")
        self.historyHDF = f"train_{self.exec}.hdf"
        self.logDir = f"runs/experiment_{self.exec}"
        if self.batchSize % self.batchSplit :
            raise Exception(f"Batch size {self.batchSize} is not divisible by batch split {self.batchSplit}.")
global TCfg
TCfg = initIfNew('TCfg')


@dataclass
class DCfgClass:
    inShape : tuple = (80,80)
DCfg = initIfNew('DCfg')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def plotData(dataY, rangeY=None, dataYR=None, rangeYR=None,
             dataX=None, rangeX=None, rangeP=None,
             figsize=(16,8), saveTo=None, show=True):

    if type(dataY) is np.ndarray :
        plotData((dataY,), rangeY=rangeY, dataYR=dataYR, rangeYR=rangeYR,
             dataX=dataX, rangeX=rangeX, rangeP=rangeP,
             figsize=figsize, saveTo=saveTo, show=show)
        return
    if type(dataYR) is np.ndarray :
        plotData(dataY, rangeY=rangeY, dataYR=(dataYR,), rangeYR=rangeYR,
             dataX=dataX, rangeX=rangeX, rangeP=rangeP,
             figsize=figsize, saveTo=saveTo, show=show)
        return
    if type(dataY) is not tuple :
        eprint(f"Unknown data type to plot: {type(dataY)}.")
        return
    if type(dataYR) is not tuple and dataYR is not None:
        eprint(f"Unknown data type to plot: {type(dataYR)}.")
        return

    last = min( len(data) for data in dataY )
    if dataYR is not None:
        last = min( last,  min( len(data) for data in dataYR ) )
    if dataX is not None:
        last = min(last, len(dataX))
    if rangeP is None :
        rangeP = (0,last)
    elif type(rangeP) is int :
        rangeP = (0,rangeP) if rangeP > 0 else (-rangeP,last)
    elif type(rangeP) is tuple :
        rangeP = ( 0    if rangeP[0] is None else rangeP[0],
                   last if rangeP[1] is None else rangeP[1],)
    else :
        eprint(f"Bad data type on plotData input rangeP: {type(rangeP)}")
        raise Exception(f"Bug in the code.")
    rangeP = np.s_[ max(0, rangeP[0]) : min(last, rangeP[1]) ]
    if dataX is None :
        dataX = np.arange(rangeP.start, rangeP.stop)

    plt.style.use('default')
    plt.style.use('dark_background')
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.xaxis.grid(True, 'both', linestyle='dotted')
    if rangeX is not None :
        ax1.set_xlim(rangeX)
    else :
        ax1.set_xlim(rangeP.start,rangeP.stop-1)

    ax1.yaxis.grid(True, 'both', linestyle='dotted')
    nofPlots = len(dataY)
    if rangeY is not None:
        ax1.set_ylim(rangeY)
    colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
    for idx , data in enumerate(dataY):
        ax1.plot(dataX, data[rangeP], linestyle='-',  color=colors[idx])

    if dataYR is not None : # right Y axis
        ax2 = ax1.twinx()
        ax2.yaxis.grid(True, 'both', linestyle='dotted')
        nofPlots = len(dataYR)
        if rangeYR is not None:
            ax2.set_ylim(rangeYR)
        colors = [ matplotlib.colors.hsv_to_rgb((hv/nofPlots, 1, 1)) for hv in range(nofPlots) ]
        for idx , data in enumerate(dataYR):
            ax2.plot(dataX, data[rangeP], linestyle='dashed',  color=colors[idx])

    if saveTo:
        fig.savefig(saveTo)
    if not show:
        plt.close(fig)


def plotImage(image) :
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    plt.show()


def plotImages(images) :
    for i, img in enumerate(images) :
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.axis("off")
    plt.show()


def sliceShape(shape, sl) :
    if type(shape) is int :
        shape = torch.Size([shape])
    if type(sl) is tuple :
        if len(shape) != len(sl) :
            raise Exception(f"Different sizes of shape {shape} and sl {sl}")
        out = []
        for i in range(0, len(shape)) :
            indeces = sl[i].indices(shape[i])
            out.append(indeces[1]-indeces[0])
        return out
    elif type(sl) is slice :
        indeces = sl.indices(shape[0])
        return indeces[1]-indeces[0]
    else :
        raise Exception(f"Incompatible object {sl}")


def fillWheights(seq) :
    for wh in seq :
        if hasattr(wh, 'weight') :
            #torch.nn.init.xavier_uniform_(wh.weight)
            #torch.nn.init.zeros_(wh.weight)
            #torch.nn.init.constant_(wh.weight, 0)
            #torch.nn.init.uniform_(wh.weight, a=0.0, b=1.0, generator=None)
            torch.nn.init.normal_(wh.weight, mean=0.0, std=0.01)
        if hasattr(wh, 'bias') :
            torch.nn.init.normal_(wh.bias, mean=0.0, std=0.01)



def set_seed(SEED_VALUE):
    torch.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed(SEED_VALUE)
    torch.cuda.manual_seed_all(SEED_VALUE)
    np.random.seed(SEED_VALUE)


def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    return


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path, map_location=TCfg.device))
    return model



def addToHDF(filename, containername, data) :
    if len(data.shape) == 2 :
        data=np.expand_dims(data, 0)
    if len(data.shape) != 3 :
        raise Exception(f"Not appropriate input array size {data.shape}.")

    with h5py.File(filename,'a') as file :

        if  containername not in file.keys():
            dset = file.create_dataset(containername, data.shape,
                                       maxshape=(None,data.shape[1],data.shape[2]),
                                       dtype='f')
            dset[()] = data
            return

        dset = file[containername]
        csh = dset.shape
        if csh[1] != data.shape[1] or csh[2] != data.shape[2] :
            raise Exception(f"Shape mismatch: input {data.shape}, file {dset.shape}.")
        msh = dset.maxshape
        newLen = csh[0] + data.shape[0]
        if msh[0] is None or msh[0] >= newLen :
            dset.resize(newLen, axis=0)
        else :
            raise Exception(f"Insufficient maximum shape {msh} to add data"
                            f" {data.shape} to current volume {dset.shape}.")
        dset[csh[0]:newLen,...] = data
        file.close()


    return 0


def createWriter(logDir, addToExisting=False) :
    if not addToExisting and os.path.exists(logDir) :
        raise Exception(f"Log directory \"{logDir}\" for the experiment already exists."
                        " Remove it or implicitry overwrite with setting addToExisting to True.")
    return SummaryWriter(logDir)
writer = initIfNew('writer')


def loadImage(imageName, expectedShape=None) :
    if not imageName:
        return None
    #imdata = imread(imageName).astype(np.float32)
    imdata = tifffile.imread(imageName).astype(np.float32)
    if len(imdata.shape) == 3 :
        imdata = np.mean(imdata[:,:,0:3], 2)
    if not expectedShape is None  and  imdata.shape != expectedShape :
        raise Exception(f"Dimensions of the input image \"{imageName}\" {imdata.shape} "
                        f"do not match expected shape {expectedShape}.")
    return imdata


class SamplingMask :
    def __init__(self):
        self.mask = loadImage("sampling_mask.tif")
        mn = self.mask.min()
        mx = self.mask.max()
        if mn < mx :
            self.mask = (self.mask - mn) / (mx - mn)
        else :
            self.mask = 1
        imask = fn.conv2d(torch.from_numpy(self.mask)[None, None,...],
                          torch.ones((1, 1,*DCfg.inShape)))[0,0,...].numpy() / math.prod(DCfg.inShape)
        #imask = np.convolve(self.mask, kernel, mode='valid')
        self.indicies = np.argwhere( np.logical_and(imask < 0.9, imask > 0.5) ).astype(int)
    def __len__(self) :
        return len(self.indicies)
    def __getitem__(self, index=None) :
        if index is None :
            index = random.randint(0, self.__len__()-1)
        else:
            index = index % self.__len__()
        idx = self.indicies[index]
        return self.mask[idx[0]:idx[0]+DCfg.inShape[-2], idx[1]:idx[1]+DCfg.inShape[-1]]

samplingMask = initIfNew('samplingMask')


class ShiftedPair :

    def hdfData(self,inputString):
        nameSplit = inputString.split(':')
        if len(nameSplit) != 2 :
            raise Exception(f"String \"{inputString}\" does not represent an HDF5 format \"fileName:container\".")
        hdfName = nameSplit[0]
        hdfVolume = nameSplit[1]
        try :
            trgH5F =  h5py.File(hdfName,'r')
        except :
            raise Exception(f"Failed to open HDF file '{hdfName}'.")
        if  hdfVolume not in trgH5F.keys():
            raise Exception(f"No dataset '{hdfVolume}' in input file {hdfName}.")
        data = trgH5F[hdfVolume]
        if not data.size :
            raise Exception(f"Container \"{inputString}\" is zero size.")
        sh = data.shape
        if len(sh) != 3 :
            raise Exception(f"Dimensions of the container \"{inputString}\" is not 3: {sh}.")
        return data


    def __init__(self, orgVol, sftVol, orgMask=None, sftMask=None) :
        self.orgData = self.hdfData(orgVol)
        self.shape = self.orgData.shape
        self.face = self.shape[1:]
        self.imgs = self.shape[0]
        self.sftData = self.hdfData(sftVol)
        if self.sftData.shape != self.shape :
            raise Exception(f"Shape mismatch of shifted volume: {self.sftData.shape} != {self.shape}.")
        def cookMask(maskName) :
            if maskName is None:
                mask =  np.ones(self.face)
            else :
                mask = loadImage(maskName, self.face)
                mn = mask.min()
                mx = mask.max()
                if mn < mx :
                    mask = (mask - mn) / (mx - mn)
                else :
                    mask = np.ones(self.face)
            imask = fn.conv2d(torch.from_numpy(mask)[None, None,...],
                              torch.ones((1, 1,*DCfg.inShape)))[0,0,...].numpy() / math.prod(DCfg.inShape)
            return mask, imask
        self.orgMask, self.orgImask = cookMask(orgMask)
        self.sftMask, self.sftImask = cookMask(sftMask)
        self.iFace = self.orgImask.shape
        self.goodForTraining = np.argwhere( np.logical_or(self.orgImask > 0.75,
                                                          self.sftImask > 0.75) ).astype(int)
        self.prehash = hash((orgVol, sftVol, orgMask, sftMask))

    def __len__(self):
        return len(self.goodForTraining) * self.imgs


    def __getitem__(self, index=None) :
        if type(index) is tuple and len(index) == 3 :
            zdx, ydx, xdx = index
        elif type(index) is int :
            if index >= self.__len__() :
                raise Exception(f"Index {index} is out of range for shifted pair"
                                f" of size {self.__len__()}.")
            zdx = index // len(self.goodForTraining)
            ydx, xdx = self.goodForTraining[ index % len(self.goodForTraining) ]
        elif index is None :
            zdx = random.randint(0,self.imgs-1)
            ydx, xdx = self.goodForTraining[ random.randint(0,len(self.goodForTraining)-1) ]
            #while True:
            #    ydx = random.randint(0,self.iFace[-2]-1)
            #    xdx = random.randint(0,self.iFace[-1]-1)
            #    range = np.s_[ydx:ydx+DCfg.inShape[-2], xdx:xdx+DCfg.inShape[-1]]
            #    orgSubMask = self.orgMask[range]
            #    sftSubMask = self.sftMask[range]
            #    if self.orgIMask[ydx,xdx] > 0.75 and \
            #       self.sftIMask[ydx,xdx] > 0.75 and \
            #       np.count_nonzero(orgSubMask + sftSubMask) / math.prod(DCfg.inShape) > 0.5 :
            #        break
        else :
            raise Exception(f"Bad index type {type(index)} {index} shifted pair.")
        range = np.s_[ydx:ydx+DCfg.inShape[-2], xdx:xdx+DCfg.inShape[-1]]
        orgSubMask = self.orgMask[range]
        sftSubMask = self.sftMask[range]
        orgTrainMask = samplingMask[ hash( (0, self.prehash, zdx, ydx, xdx) ) ]
        sftTrainMask = samplingMask[ hash( (1, self.prehash, zdx, ydx, xdx) ) ]
        data = np.stack([ self.orgData[zdx, *range], self.sftData[zdx, *range],
                          orgSubMask, sftSubMask, orgTrainMask, sftTrainMask ])
        return data, (zdx, int(ydx), int(xdx))


    def masks(self) :
        return self.orgMask, self.sftMask



class ManyShiftedPairs :

    def __init__(self, listOfPairs) :
        self.pairs = []
        for pair in listOfPairs :
            self.pairs.append(ShiftedPair(*pair))

    def __len__(self):
        return sum( [ len(pair) for pair in self.pairs ] )

    def __getitem__(self, index=None):

        if type(index) is tuple and len(index) == 4 :
            return self.pairs[index[0]].__getitem__(index[1:])[0], index
        elif type(index) is int :
            if index >= self.__len__() :
                raise Exception(f"Index {index} is out of range for collection of length {self.__len__()}.")
            tail = index
            curPair = 0
            while tail >= len(self.pairs[curPair]) and curPair < len(self.pairs) :
                tail -= len(self.pairs[curPair])
                curPair += 1
            data, index = self.pairs[curPair].__getitem__(tail)
            return data, (curPair, *index)
        elif index is None :
            return self.__getitem__( random.randint(0,self.__len__()-1) )
        else :
            raise Exception(f"Bad index type {type(index)} {index} for collection of pairs.")


    def get_dataset(self, transform=None) :

        class InputFromPairs(torch.utils.data.Dataset) :
            def __init__(self, root, transform=None):
                self.container = root
                #self.oblTransform = transforms.Compose( [ transforms.ToTensor() ] )
                self.transform = transform
            def __len__(self):
                return self.container.__len__()
            def __getitem__(self, index=None, doTransform=True):
                data, index = self.container.__getitem__(index)
                #data = self.oblTransform(data)
                data = torch.tensor(data, device=TCfg.device)
                if doTransform and self.transform :
                    data = self.transform(data)
                return (data, index)


        return InputFromPairs(self, transform)


dataRoot = "/mnt/hddData/shiftpatch/"
TestShiftedPairs = [ [ dataRoot + prefix + postfix
                       for postfix in ["_org.hdf:/data",
                                       "_sft.hdf:/data",
                                       "_org_mask.tif",
                                       "_sft_mask.tif"] ]
                         for prefix in [ "01_dir", "01_flp" ] ]
TrainShiftedPairs = [
    ( "org.hdf:/data", "sft.hdf:/data", "sft_mask.tif", "org_mask.tif" ) ,
]




examplesDb = []
examples = initIfNew('examples')


def createTrainSet() :
    setRoot = ManyShiftedPairs(TrainShiftedPairs)
    mytransforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    return setRoot.get_dataset(mytransforms)


def createTestSet() :
    setRoot = ManyShiftedPairs(TestShiftedPairs)
    mytransforms = transforms.Compose([
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    return setRoot.get_dataset(mytransforms)



def createDataLoader(tSet, num_workers=os.cpu_count()) :
    return torch.utils.data.DataLoader(
        dataset=tSet,
        batch_size=TCfg.batchSize,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )



def createReferences(tSet, toShow = 0) :
    global examples
    examples = examplesDb.copy()
    if toShow :
        examples.insert(0, examples.pop(toShow))
    mytransforms = transforms.Compose([
            transforms.Normalize(mean=(0.5), std=(1))
    ])
    refImages = torch.stack( [ mytransforms(tSet.__getitem__(ex, doTransform=False)[0])
                               for ex in examples ] ).to(TCfg.device)
    refNoises = torch.randn((refImages.shape[0],TCfg.latentDim)).to(TCfg.device)
    return refImages, refNoises
refImages = initIfNew('refImages')
refNoises = initIfNew('refNoises')


def showMe(tSet, item=None) :
    global refImages, refNoises
    image = None
    if item is None :
        while True:
            image, index = tSet[random.randint(0,len(tSet)-1)]
            if image[0].mean() > 0 and image[0].min() < -0.1 :
                print (f"{index}")
                break
    elif isinstance(item, int) :
        image = refImages[0,...]
    else :
        image, _ = tSet.__getitem__(item)
    image = image.squeeze()
    plotImages( [image[0].cpu(), image[2].cpu(), image[4].cpu()] )
    plotImages( [image[1].cpu(), image[3].cpu(), image[5].cpu()] )
    image = image.to(TCfg.device)


save_interim = None


class GeneratorTemplate(nn.Module):

    def __init__(self, latentChannels=0):
        super(GeneratorTemplate, self).__init__()
        self.latentChannels = latentChannels
        self.baseChannels = 64
        self.amplitude = 4


    def createLatent(self) :
        if self.latentChannels == 0 :
            return None
        toRet =  nn.Sequential(
            nn.Linear(TCfg.latentDim, self.sinoSize*self.latentChannels),
            nn.ReLU(),
            nn.Unflatten( 1, (self.latentChannels,) + self.sinoSh )
        )
        fillWheights(toRet)
        return toRet


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def decblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True,
                                          padding=1) \
                       if stride == 1 and dopadding else \
                       nn.ConvTranspose2d(chIn, chOut, kernel, stride, bias=True)
                      )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)


    def createFClink(self) :
        smpl = torch.zeros((1, 4+self.latentChannels, *self.sinoSh))
        with torch.no_grad() :
            for encoder in self.encoders :
                smpl = encoder(smpl)
        encSh = smpl.shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Linear(linChannels, linChannels),
            nn.LeakyReLU(0.2),
            nn.Unflatten(1, encSh[1:]),
        )
        fillWheights(toRet)
        return toRet


    def createLastTouch(self) :
        toRet = nn.Sequential(
            nn.Conv2d(self.baseChannels+4, 2, 1),
            nn.Tanh(),
        )
        fillWheights(toRet)
        return toRet

#    def fillImages(self, images, noises=None) :
#        images[self.gapRng] = self.generatePatches(images, noises)
#        return images


#    def generateImages(self, images, noises=None) :
#        clone = images.clone()
#        return self.fillImages(clone, noises)


    def forward(self, input):

        if not save_interim is None :
            save_interim = {}
        def saveToInterim(key, data) :
            if not save_interim is None :
                save_interim[key] = data.clone().detach()

        images, noises = input
        saveToInterim('input', images )
        orgDims = images.dim()
        if orgDims == 3 :
            images = images.view(1, *images.shape)
        modelIn = images.clone().detach()

        if self.latentChannels :
            latent = self.noise2latent(noises)
            dwTrain = [torch.cat((modelIn, latent), dim=1),]
        else :
            dwTrain = [modelIn,]
        for encoder in self.encoders :
            dwTrain.append(encoder(dwTrain[-1]))
        mid = self.fcLink(dwTrain[-1])
        upTrain = [mid]
        for level, decoder in enumerate(self.decoders) :
            upTrain.append( decoder( torch.cat( (upTrain[-1], dwTrain[-1-level]), dim=1 ) ) )
        res = self.lastTouch(torch.cat( (upTrain[-1], modelIn ), dim=1 ))
        if orgDims == 3 :
            res = res.squeeze(0)
        saveToInterim('output', res)
        return res


generator = initIfNew('generator')
lowResGenerators = initIfNew('lowResGenerators', {})


class DiscriminatorTemplate(nn.Module):

    def __init__(self, omitEdges=0):
        super(DiscriminatorTemplate, self).__init__()
        self.baseChannels = 64
        self.omitEdges = omitEdges


    def encblock(self, chIn, chOut, kernel, stride=1, norm=False, dopadding=False) :
        chIn = int(chIn*self.baseChannels)
        chOut = int(chOut*self.baseChannels)
        layers = []
        layers.append( nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True,
                                padding='same', padding_mode='reflect') \
                                if stride == 1 and dopadding else \
                                nn.Conv2d(chIn, chOut, kernel, stride=stride, bias=True)
                     )
        if norm :
            layers.append(nn.BatchNorm2d(chOut))
        layers.append(nn.LeakyReLU(0.2))
        fillWheights(layers)
        return torch.nn.Sequential(*layers)

    def createHead(self) :
        encSh = self.body(torch.zeros((1,2,*DCfg.inShape))).shape
        linChannels = math.prod(encSh)
        toRet = nn.Sequential(
            nn.Flatten(),
            #nn.Dropout(0.4),
            nn.Linear(linChannels, self.baseChannels*4),
            #nn.Linear(linChannels, 1),
            nn.LeakyReLU(0.2),
            #nn.Dropout(0.4),
            nn.Linear(self.baseChannels*4, 1),
            nn.Sigmoid(),
        )
        fillWheights(toRet)
        return toRet

    def forward(self, images):
        if images.dim() == 3:
            images = images.unsqueeze(1)
        if self.omitEdges :
            images = images.clone() # I want to exclude two blocks on the edges :
            images[ ..., :self.omitEdges, DCfg.gapRngX ] = 0
            images[ ..., -self.omitEdges:, DCfg.gapRngX ] = 0
        convRes = self.body(images)
        res = self.head(convRes)
        return res

discriminator = initIfNew('discriminator')
noAdv=False


def createOptimizer(model, lr) :
    return optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.5, 0.999)
    )
optimizer_G = initIfNew('optimizer_G')
optimizer_D = initIfNew('optimizer_D')


def restoreCheckpoint(path=None, logDir=None) :
    if logDir is None :
        logDir = TCfg.logDir
    if path is None :
        if os.path.exists(logDir) :
            raise Exception(f"Starting new experiment with existing log directory \"{logDir}\"."
                            " Remove it .")
        try : os.remove(TCfg.historyHDF)
        except : pass
        return 0, 0, 0, 1, 0, TrainResClass()
    else :
        return loadCheckPoint(path, generator, discriminator, optimizer_G, optimizer_D)


def saveModels(path="") :
    save_model(generator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_gen.pt" )
    save_model(discriminator, model_path = ( path if path else f"model_{TCfg.exec}" ) + "_dis.pt"  )


def createCriteria() :
    BCE = nn.BCELoss(reduction='none')
    MSE = nn.MSELoss(reduction='none')
    L1L = nn.L1Loss(reduction='none')
    return BCE, MSE, L1L
BCE, MSE, L1L = createCriteria()
lossDifCoef = 0
lossAdvCoef = 1.0

#def applyWeights(inp, weights, storePerIm=None):
#    inp = inp.squeeze()
#    if not inp.dim() :
#        inp = inp.unsqueeze(0)
#    sum = len(inp)
#    if not weights is None :
#        inp *= weights
#        sum = weights.sum()
#    if storePerIm is not None : # must be list
#        storePerIm.extend(inp.tolist())
#    return inp.sum()/sum

def loss_Adv(y_true, y_pred, weights=None, storePerIm=None):
    return BCE(y_pred, y_true)

def loss_MSE(p_true, p_pred):
    mse = MSE(p_true[:,0,...], p_pred[:,0,...]) * (1-p_true[:,2,...]) * p_true[:,3,...] \
        + MSE(p_true[:,1,...], p_pred[:,1,...]) * (1-p_true[:,3,...]) * p_true[:,2,...]
    return mse.sum()#dim=(-1,-2))

def loss_L1L(p_true, p_pred):
    mse = L1L(p_true[:,0,...], p_pred[:,0,...]) * (1-p_true[:,2,...]) * p_true[:,3,...] \
        + L1L(p_true[:,1,...], p_pred[:,1,...]) * (1-p_true[:,3,...]) * p_true[:,2,...]
    return mse.sum()#dim=(-1,-2))

def loss_Rec(p_true, p_pred):
    return loss_MSE(p_true, p_pred)

def loss_Gen(y_true, y_pred, p_true, p_pred):
    lossAdv = loss_Adv(y_true, y_pred)
    lossDif = loss_Rec(p_pred, p_true)
    return lossAdv, lossDif


def summarizeSet(dataloader, onPrep=True):

    MSE_diffs, L1L_diffs, Rec_diffs = [], [], []
    Real_probs, Fake_probs, GA_losses, GD_losses, D_losses = [], [], [], [], []
    totalNofIm = 0
    generator.to(TCfg.device)
    generator.eval()
    discriminator.eval()
    with torch.no_grad() :
        for it , data in tqdm.tqdm(enumerate(dataloader), total=int(len(dataloader))):
            images = data[0].squeeze(1).to(TCfg.device)
            nofIm = images.shape[0]
            subBatchSize = nofIm // TCfg.batchSplit
            totalNofIm += nofIm
            procImages, procData = imagesPreProc(images)
            genImages = procImages.clone()
            fprobs = torch.zeros((nofIm,1), device=TCfg.device)
            rprobs = torch.zeros((nofIm,1), device=TCfg.device)

            rprob = fprob = 0
            for i in range(TCfg.batchSplit) :
                subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
                subProcImages = procImages[subRange,...]
                patchImages = generator.preProc(subProcImages) \
                              if onPrep else \
                              generator.generatePatches(subProcImages)
                genImages[subRange,:,DCfg.gapRngX] = patchImages
                if not noAdv :
                    subRprobs = discriminator(subProcImages)
                    #if storesPerIm[3] is not None :
                    #    storesPerIm[3].extend(rprobs.tolist())
                    rprobs[subRange,...] = subRprobs
                    rprob += subRprobs.sum().item()
                    subFprobs = discriminator(genImages[subRange,...])
                    #if storesPerIm[4] is not None :
                    #    storesPerIm[4].extend(fprobs.tolist())
                    fprobs[subRange,...] = subFprobs
                    fprob += subFprobs.sum().item()
            procImages = imagesPostProc(genImages, procData)
            MSE_diffs.append( nofIm * loss_MSE(images[DCfg.gapRng], procImages[DCfg.gapRng]
                                              ,storePerIm = storesPerIm[2]))
            L1L_diffs.append( nofIm * loss_L1L(images[DCfg.gapRng], procImages[DCfg.gapRng]
                                              ,storePerIm = storesPerIm[1]))
            Rec_diffs.append( nofIm * loss_Rec(images[DCfg.gapRng], procImages[DCfg.gapRng]
                                              ,storePerIm = storesPerIm[0]
                                              ,normalizeRec=calculateNorm(images)))
            if not noAdv :
                labelsTrue = torch.full((nofIm, 1),  1 - TCfg.labelSmoothFac,
                            dtype=torch.float, device=TCfg.device, requires_grad=False)
                labelsFalse = torch.full((nofIm, 1),  TCfg.labelSmoothFac,
                            dtype=torch.float, device=TCfg.device, requires_grad=False)
                labelsDis = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device).requires_grad_(False)
                subD_loss = loss_Adv(labelsDis, torch.cat((rprobs, fprobs), dim=0))
                subGA_loss, subGD_loss = loss_Gen(labelsTrue, fprobs,
                                                  images[DCfg.gapRng], procImages[DCfg.gapRng],
                                                  normalizeRec=calculateNorm(images))
                D_losses.append( nofIm * subD_loss )
                GA_losses.append( nofIm * subGA_loss )
                GD_losses.append( nofIm * subGD_loss )
                Real_probs.append(rprob)
                Fake_probs.append(fprob)

    MSE_diff = sum(MSE_diffs) / totalNofIm
    L1L_diff = sum(L1L_diffs) / totalNofIm
    Rec_diff = sum(Rec_diffs) / totalNofIm
    Real_prob = sum(Real_probs) / totalNofIm if not noAdv else 0
    Fake_prob = sum(Fake_probs) / totalNofIm if not noAdv else 0
    D_loss = sum(D_losses) / totalNofIm if not noAdv else 0
    GA_loss = sum(GA_losses) / totalNofIm if not noAdv else 0
    GD_loss = sum(GD_losses) / totalNofIm if not noAdv else 0

    print (f"Summary. Rec: {Rec_diff:.3e}, MSE: {MSE_diff:.3e}, L1L: {L1L_diff:.3e}, Dis: {Real_prob:.3e}, Gen: {Fake_prob:.3e}.")
    return Rec_diff, MSE_diff, L1L_diff, Real_prob, Fake_prob, D_loss, GA_loss, GD_loss

[
#def generateDiffImages(images, layout=None) :
#    images, orgDim = unsqeeze4dim(images)
#    dif = torch.zeros((images.shape[0], 1, *DCfg.sinoSh))
#    hGap = DCfg.gapW // 2
#    pre = images.clone()
#    gen = images.clone()
#    with torch.no_grad() :
#        wghts = calculateWeights(images)
#        generator.eval()
#        pre[DCfg.gapRng] = generator.preProc(images)
#        gen[DCfg.gapRng] = generator.generatePatches(images)
#        dif[DCfg.gapRng] = (gen - pre)[DCfg.gapRng]
#        dif[...,hGap:hGap+DCfg.gapW] = (images - pre)[DCfg.gapRng]
#        dif[...,-DCfg.gapW-hGap:-hGap] = (images - gen)[DCfg.gapRng]
#        for curim in range(images.shape[0]) :
#            if ( cof := max(-dif[curim,...].min(),dif[curim,...].max()) ) != 0 :
#                dif[curim,...] /= cof
#            else :
#                dif[curim,...] = 0
#        probs = torch.empty(images.shape[0],3)
#        dists = torch.empty(images.shape[0],3)
#        #discriminator.eval()
#        probs[:,0] = discriminator(images)[:,0]
#        probs[:,1] = discriminator(pre)[:,0]
#        probs[:,2] = discriminator(gen)[:,0]
#        dists[:,0] = loss_Rec(images[DCfg.gapRng], gen[DCfg.gapRng], wghts, normalizeRec=calculateNorm(images))
#        dists[:,1] = loss_MSE(images[DCfg.gapRng], gen[DCfg.gapRng], wghts)
#        dists[:,2] = loss_L1L(images[DCfg.gapRng], gen[DCfg.gapRng], wghts)
#
#    simages = None
#    if not layout is None :
#        def stretch(stretchme, mm, aa) :
#            return ( stretchme - mm ) * 2 / aa - 1 if ampl > 0 else stretchme * 0
#        simages = images.clone()
#        for curim in range(images.shape[0]) :
#            rng = np.s_[curim,...]
#            minv = min(images[rng].min(), pre[rng].min(), gen[rng].min()).item()
#            ampl = max(images[rng].max(), pre[rng].max(), gen[rng].max()).item() - minv
#            simages[rng] = stretch(simages[rng], minv, ampl)
#            pre[rng] = stretch(pre[rng], minv, ampl)
#            gen[rng] = stretch(gen[rng], minv, ampl)
#
#    cGap = DCfg.gapW
#    if layout == 0 :
#        collage = torch.empty(images.shape[0], 4, *DCfg.sinoSh)
#        collage[:,0,...] = simages[:,0,...]
#        collage[:,1,...] = pre[:,0,...]
#        collage[:,2,...] = gen[:,0,...]
#        collage[:,3,...] = dif[:,0,...]
#    elif layout == 2 :
#        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0]*2 + cGap, DCfg.sinoSh[1]*2 + cGap ))
#        collage[..., :DCfg.sinoSh[0], :DCfg.sinoSh[1]] = gen
#        collage[..., :DCfg.sinoSh[0], DCfg.sinoSh[1]+cGap:] = pre
#        collage[..., DCfg.sinoSh[0]+cGap:, :DCfg.sinoSh[1]] = simages
#        collage[..., DCfg.sinoSh[0]+cGap:, DCfg.sinoSh[1]+cGap:] = dif
#    elif layout == 4 :
#        collage = torch.zeros((images.shape[0], 1, DCfg.sinoSh[0], 4*DCfg.sinoSh[1] + 3*cGap))
#        collage[..., :DCfg.sinoSh[1]] = simages
#        collage[..., DCfg.sinoSh[1]+cGap:2*DCfg.sinoSh[1]+cGap] = gen
#        collage[..., 2*DCfg.sinoSh[1]+2*cGap:3*DCfg.sinoSh[1]+2*cGap] = dif
#        collage[..., 3*DCfg.sinoSh[1]+3*cGap:4*DCfg.sinoSh[1]+4*cGap] = pre
#    elif layout == -4 :
#        collage = torch.zeros( (images.shape[0], 1, 4*DCfg.sinoSh[0] + 3*cGap, DCfg.sinoSh[1]))
#        collage[... , :DCfg.sinoSh[0] , : ] = simages
#        collage[... , DCfg.sinoSh[0]+cGap:2*DCfg.sinoSh[0]+cGap , :] = gen
#        collage[... , 2*DCfg.sinoSh[0]+2*cGap:3*DCfg.sinoSh[0]+2*cGap , : ] = dif
#        collage[... , 3*DCfg.sinoSh[0]+3*cGap:4*DCfg.sinoSh[0]+4*cGap , : ] = pre
#    else :
#        collage = dif
#    collage = squeezeOrg(collage,orgDim)
#
#    return collage, probs, dists
#
#
#def logStep(iter, write=True) :
#    colImgs, probs, dists = generateDiffImages(refImages, layout=-4)
#    probs = probs.mean(dim=0)
#    dists = dists.mean(dim=0)
#    colImgs = colImgs.squeeze()
#    cSh = colImgs.shape
#    gapH = DCfg.gapW
#    collage = np.zeros( ( cSh[-2], cSh[0]*cSh[-1] + (cSh[0]-1)*gapH ), dtype=np.float32  )
#    for curI in range(cSh[0]) :
#        collage[ : , curI * (cSh[-1]+gapH) : curI * (cSh[-1]+gapH) + cSh[-1]] = colImgs[curI,...]
#    #writer.add_scalars("Probs of ref images",
#    #                   {'Ref':probs[0]
#    #                   ,'Gen':probs[2]
#    #                   ,'Pre':probs[1]
#    #                   }, iter )
#    #writer.add_scalars("Dist of ref images",
#    #                   { 'REC' : dists[0]
#    #                   , 'MSE' : dists[1]
#    #                   , 'L1L' : dists[2]
#    #                   }, iter )
#    try :
#        addToHDF(TCfg.historyHDF, "data", collage)
#    except :
#        eprint("Failed to save.")
#    return collage, probs, dists


#def initialTest() :
#    with torch.inference_mode() :
#        collage, probs, _ = logStep(iter, not iter)
#        print("Probabilities of reference images: "
#              f'Ref: {probs[0]:.3e}, '
#              f'Gen: {probs[2]:.3e}, '
#              f'Pre: {probs[1]:.3e}.')
#        #generator.eval()
#        pre = generator.preProc(refImages)
#        wghts = calculateWeights(refImages)
#        ref_loss_Rec = loss_Rec(refImages[DCfg.gapRng], pre, wghts, normalizeRec=calculateNorm(refImages))
#        ref_loss_MSE = loss_MSE(refImages[DCfg.gapRng], pre, wghts)
#        ref_loss_L1L = loss_L1L(refImages[DCfg.gapRng], pre, wghts)
#        print("Distances of reference images: "
#              f"REC: {ref_loss_Rec:.3e}, "
#              f"MSE: {ref_loss_MSE:.3e}, "
#              f"L1L: {ref_loss_L1L:.3e}.")
#        #if not epoch :
#        #    writer.add_scalars("Dist of ref images",
#        #                          { 'REC' : ref_loss_Rec
#        #                          , 'MSE' : ref_loss_MSE
#        #                          , 'L1L' : ref_loss_L1L
#        #                          }, 0 )
#        plotImage(collage)
]


def calculateWeights(images) :
    return None

def calculateNorm(images) :
    mean2 = images[...,:DCfg.gapRngX.start].mean(dim=(-1,-2)) \
          + images[...,DCfg.gapRngX.stop:].mean(dim=(-1,-2))
    return 2 / ( 1 + mean2 + 1e-5 ) # to denorm and adjust for mean



def imagesPreProc(images) :
    return images, None

def imagesPostProc(images, procData=None) :
    return images


@dataclass
class TrainInfoClass:
    testIndex = 0
    orgImage = None
    sftImage = None
    orgRecImage = None
    sftRecImage = None
    orgMask = None
    sftMask = None

@dataclass
class TrainResClass:
    lossD : any = 0
    lossGA : any = 0
    lossGD : any = 0
    lossMSE : any = 0
    lossL1L : any = 0
    predReal : any = 0
    predPre : any = 0
    predFake : any = 0
    nofIm : int = 0
    def __add__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            setattr(toRet, fn, getattr(self, fn) + getattr(other, fn) )
        return toRet
    def __mul__(self, other):
        toRet = TrainResClass()
        for field in dataclasses.fields(TrainResClass):
            fn = field.name
            setattr(toRet, fn, getattr(self, fn) * other )
        return toRet
    __rmul__ = __mul__




def saveCheckPoint(path, epoch, iterations, minGEpoch, minGdLoss,
                   generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None,
                   startFrom=0, interimRes=TrainResClass()) :
    checkPoint = {}
    checkPoint['epoch'] = epoch
    checkPoint['iterations'] = iterations
    checkPoint['minGEpoch'] = minGEpoch
    checkPoint['minGdLoss'] = minGdLoss
    checkPoint['startFrom'] = startFrom
    checkPoint['generator'] = generator.state_dict()
    checkPoint['discriminator'] = discriminator.state_dict()
    if not optimizerGen is None :
        checkPoint['optimizerGen'] = optimizerGen.state_dict()
    if not schedulerGen is None :
        checkPoint['schedulerGen'] = schedulerGen.state_dict()
    if not optimizerDis is None :
        checkPoint['optimizerDis'] = optimizerDis.state_dict()
    if not schedulerDis is None :
        checkPoint['schedulerDis'] = schedulerDis.state_dict()
    checkPoint['resAcc'] = interimRes
    torch.save(checkPoint, path)


def loadCheckPoint(path, generator, discriminator,
                   optimizerGen=None, optimizerDis=None,
                   schedulerGen=None, schedulerDis=None) :
    checkPoint = torch.load(path, map_location=TCfg.device, weights_only=False)
    epoch = checkPoint['epoch']
    iterations = checkPoint['iterations']
    minGEpoch = checkPoint['minGEpoch']
    minGdLoss = checkPoint['minGdLoss']
    startFrom = checkPoint['startFrom'] if 'startFrom' in checkPoint else 0
    generator.load_state_dict(checkPoint['generator'])
    discriminator.load_state_dict(checkPoint['discriminator'])
    if not optimizerGen is None :
        optimizerGen.load_state_dict(checkPoint['optimizerGen'])
    if not schedulerGen is None :
        schedulerGen.load_state_dict(checkPoint['schedulerGen'])
    if not optimizerDis is None :
        optimizerDis.load_state_dict(checkPoint['optimizerDis'])
    if not schedulerDis is None :
        schedulerDis.load_state_dict(checkPoint['schedulerDis'])
    interimRes = checkPoint['resAcc'] if 'resAcc' in checkPoint else TrainResClass()

    return epoch, iterations, minGEpoch, minGdLoss, startFrom, interimRes



trainInfo = TrainInfoClass()
normMSE=1
normL1L=1
normRec=1
skipDis = False

def train_step(images):
    global trainDis, trainGen, eDinfo, noAdv, withNoGrad, skipGen, skipDis
    trainInfo.iterations += 1
    trainInfo.totPerformed += 1
    trainRes = TrainResClass()

    nofIm = images.shape[0]
    images = images.to(TCfg.device)
    procImages, procReverseData = imagesPreProc(images)
    fakeImages = procImages.clone().detach().requires_grad_(False)
    subBatchSize = nofIm // TCfg.batchSplit
    normDiff = calculateNorm(images)

    labelsTrue = torch.full((subBatchSize, 1),  1 - TCfg.labelSmoothFac,
                        dtype=torch.float, device=TCfg.device, requires_grad=False)
    labelsFalse = torch.full((subBatchSize, 1),  TCfg.labelSmoothFac,
                        dtype=torch.float, device=TCfg.device, requires_grad=False)
    labelsDis = torch.cat( (labelsTrue, labelsFalse), dim=0).to(TCfg.device).requires_grad_(False)

    # train discriminator
    if not noAdv :

        # calculate predictions of prefilled images - purely for metrics purposes
        #discriminator.eval()
        #generator.eval()
        #trainRes.predPre = 0
        #with torch.no_grad() :
        #    for i in range(TCfg.batchSplit) :
        #        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
        #        fakeImages[subRange,:,DCfg.gapRngX] = generator.preProc(procImages[subRange,...])
        #        trainRes.predPre += discriminator(fakeImages[subRange,...]).mean().item()
        #    trainRes.predPre /= TCfg.batchSplit

        pred_real = torch.empty((nofIm,1), requires_grad=False)
        pred_fake = torch.empty((nofIm,1), requires_grad=False)
        #discriminator.train()
        for param in discriminator.parameters() :
            param.requires_grad = True
        optimizer_D.zero_grad()
        for i in range(TCfg.batchSplit) :
            subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[...]
            subFakeImages = fakeImages[subRange,...]
            with torch.no_grad() :
                subFakeImages = generator.generatePatches(procImages[subRange,...])
            with torch.set_grad_enabled(not skipDis) :
                subPred_realD = discriminator(procImages[subRange,...])
                subPred_fakeD = discriminator(subFakeImages)
                pred_both = torch.cat((subPred_realD, subPred_fakeD), dim=0)
                subD_loss = loss_Adv(labelsDis, pred_both)
            # train discriminator only if it is not too good :
            if not skipDis and ( subPred_fakeD.mean() > 0.2 or subPred_realD.mean() < 0.8 ) :
                trainInfo.disPerformed += 1/TCfg.batchSplit
                subD_loss.backward()
            trainRes.lossD += subD_loss.item()
            pred_real[subRange] = subPred_realD.clone().detach()
            pred_fake[subRange] = subPred_fakeD.clone().detach()
        optimizer_D.step()
        optimizer_D.zero_grad(set_to_none=True)
        trainRes.lossD /= TCfg.batchSplit
        trainRes.predReal = pred_real.mean().item()
        trainRes.predFake = pred_fake.mean().item()

    else :
        pred_real = torch.zeros((1,), requires_grad=False)
        pred_fake = torch.zeros((1,), requires_grad=False)

    # train generator
    #discriminator.eval()
    for param in discriminator.parameters() :
        param.requires_grad = False
    #generator.train()
    optimizer_G.zero_grad()
    for i in range(TCfg.batchSplit) :
        subRange = np.s_[i*subBatchSize:(i+1)*subBatchSize] if TCfg.batchSplit > 1 else np.s_[:]
        subFakeImages = generator.generateImages(procImages[subRange,...])
        if noAdv :
            subG_loss = loss_Rec( procImages[subRange,...], subFakeImages).sum()
            subGD_loss = subGA_loss = subG_loss
        else :
            subPred_fakeG = discriminator(subFakeImages)
            subGA_loss, subGD_loss = loss_Gen(labelsTrue, subPred_fakeG,
                                              procImages[subRange,...], subFakeImages)
            subG_loss = lossAdvCoef * subGA_loss + lossDifCoef * subGD_loss
            pred_fake[subRange] = subPred_fakeG.clone().detach()
        # train generator only if it is not too good :
        #if noAdv  or  subPred_fakeD.mean() < pred_real[subRange].mean() :
        if True:
            trainInfo.genPerformed += 1/TCfg.batchSplit
            subG_loss.backward()
        trainRes.lossGA += subGA_loss.item()
        trainRes.lossGD += subGD_loss.item()
        fakeImages[subRange,...] = subFakeImages.detach()
    optimizer_G.step()
    optimizer_G.zero_grad(set_to_none=True)
    trainRes.lossGA /= TCfg.batchSplit
    trainRes.lossGD /= TCfg.batchSplit
    trainRes.predFake = pred_fake.mean().item()


    # prepare report
    with torch.no_grad() :

        trainRes.lossMSE = loss_MSE(images, fakeImages).item() / normMSE
        trainRes.lossL1L = loss_L1L(images, fakeImages).item() / normL1L
        trainRes.lossGD /= normRec

        idx = random.randint(0, nofIm-1)
        trainInfo.testIndex = idx
        trainInfo.orgImage = images[idx,0,...].clone().detach()
        trainInfo.sftImage = images[idx,1,...].clone().detach()
        trainInfo.orgRecImage = fakeImages[idx,0,...].clone().detach()
        trainInfo.sftRecImage = fakeImages[idx,1,...].clone().detach()
        trainInfo.orgMask = images[idx,2,...].clone().detach()
        trainInfo.sftMask = images[idx,3,...].clone().detach()

    return trainRes


epoch=initIfNew('epoch', 0)
iter = initIfNew('iter', 0)
imer = initIfNew('iter', 0)
minGEpoch = initIfNew('minGEpoch')
minGdLoss = initIfNew('minGdLoss', 1)
startFrom = initIfNew('startFrom', 0)

def beforeEachEpoch(epoch) :
    return

def afterEachEpoch(epoch) :
    return

def beforeReport() :
    return

def afterReport() :
    return

dataLoader=None
testLoader=None
normTestMSE=1
normTestL1L=1
normTestRec=1
normTestDis=1
normTestGen=1
normTestGDloss=1
normTestGAloss=1
normTestDloss=1
resAcc = TrainResClass()

def train(savedCheckPoint):
    global epoch, minGdLoss, minGEpoch, iter, trainInfo, startFrom, imer, resAcc
    lastGdLoss = minGdLoss
    lastGdLossTrain = 1

    discriminator.to(TCfg.device)
    generator.to(TCfg.device)
    lastUpdateTime = time.time()
    lastSaveTime = time.time()

    while TCfg.nofEpochs is None or epoch <= TCfg.nofEpochs :
        epoch += 1
        beforeEachEpoch(epoch)
        generator.train()
        discriminator.train()
        #resAcc = TrainResClass()
        totalIm = 0

        for it , data in tqdm.tqdm(enumerate(dataLoader), total=int(len(dataLoader))):
            if startFrom :
                startFrom -= 1
                continue
            iter += 1
            images = data[0].to(TCfg.device)
            nofIm = images.shape[0]
            imer += nofIm
            totalIm += nofIm
            trainRes = train_step(images)
            resAcc += trainRes * nofIm
            resAcc.nofIm += nofIm

            #if True:
            #if False :
            #if not it or it > len(dataloader)-2 or time.time() - lastUpdateTime > 60 :
            if time.time() - lastUpdateTime > 60 :
                lastUpdateTime = time.time()

                refFake = generator.generateImages(refImages)
                showMe = np.zeros( (2*DCfg.sinoSh[1] + DCfg.gapW ,
                                    5*DCfg.sinoSh[0] + 4*DCfg.gapW), dtype=np.float32  )
                for clmn in range (5) : # mark gaps
                    showMe[ DCfg.sinoSh[0] : DCfg.sinoSh[0] + DCfg.gapW ,
                            clmn*(DCfg.sinoSh[1]+DCfg.gapW) + 2*DCfg.gapW : clmn*(DCfg.sinoSh[1]+DCfg.gapW) + 3*DCfg.gapW ] = -1
                def addImage(clmn, row, img, stretch=True) :
                    imgToAdd = img.clone().detach().squeeze()
                    if stretch :
                        minv = imgToAdd.min()
                        ampl = imgToAdd.max() - minv
                        imgToAdd[()] = 2 * ( imgToAdd - minv ) / ampl - 1  if ampl!=0.0 else 0
                    showMe[ row * ( DCfg.sinoSh[1]+DCfg.gapW) : (row+1) * DCfg.sinoSh[1] + row*DCfg.gapW ,
                            clmn * ( DCfg.sinoSh[0]+DCfg.gapW) : (clmn+1) * DCfg.sinoSh[0] + clmn*DCfg.gapW ] = \
                        imgToAdd.cpu().numpy()
                addImage(0,0,trainInfo.orgImage)
                addImage(0,1,trainInfo.sftImage)
                addImage(1,0,trainInfo.orgRecImage)
                addImage(1,1,trainInfo.sftRecImage)
                addImage(2,0,trainInfo.orgMask)
                addImage(2,1,trainInfo.sftMask)
                addImage(3,0,refImages[0,0,...])
                addImage(3,1,refImages[0,1,...])
                addImage(4,0,refFake[0,0,...])
                addImage(4,1,refFake[0,1,...])
                writer.add_scalars("Losses per iter",
                                   {'Dis': trainRes.lossD
                                   ,'Gen': trainRes.lossGA
                                   ,'Rec':   lossAdvCoef * trainRes.lossGA \
                                           + lossDifCoef * trainRes.lossGD * normRec
                                   }, imer )
                writer.add_scalars("Distances per iter",
                                   {'MSE': trainRes.lossMSE
                                   ,'L1L': trainRes.lossL1L
                                   ,'REC': trainRes.lossGD
                                   }, imer )
                writer.add_scalars("Probs per iter",
                                   {'Ref':trainRes.predReal
                                   ,'Gen':trainRes.predFake
                                   ,'Pre':trainRes.predPre
                                   }, imer )

                IPython.display.clear_output(wait=True)
                beforeReport()
                print(f"Epoch: {epoch} ({minGEpoch}). " +
                      ( f" L1L: {trainRes.lossL1L:.3f} " if noAdv \
                          else \
                        f" Dis[{trainInfo.disPerformed/trainInfo.totPerformed:.2f}]: {trainRes.lossD:.3f} ({trainInfo.ratReal/trainInfo.totalImages:.3f})," ) +
                      ( f" MSE: {trainRes.lossMSE:.3f} " if noAdv \
                          else \
                        f" Gen[{trainInfo.genPerformed/trainInfo.totPerformed:.2f}]: {trainRes.lossGA:.3f} ({trainInfo.ratFake/trainInfo.totalImages:.3f})," ) +
                      f" Rec: {trainRes.lossGD:.3f} (Train: {lastGdLossTrain:.3f}, Test: {lastGdLoss/normTestRec:.3f} | {minGdLoss/normTestRec:.3f})."
                      )
                print (f"TT: {trainInfo.bestRealProb:.2f},  "
                       f"FT: {trainInfo.bestFakeProb:.2f},  "
                       f"HD: {trainInfo.highestDif/normMSE:.3e},  "
                       f"GP: {probsR[0,2].item():.3f}, {probsR[0,1].item():.3f} " )
                print (f"TF: {trainInfo.worstRealProb:.2f},  "
                       f"FF: {trainInfo.worstFakeProb:.2f},  "
                       f"LD: {trainInfo.lowestDif/normMSE:.3e},  "
                       f"R : {probsR[0,0].item():.3f}." )
                plotImage(showMe)
                afterReport()
                trainInfo = TrainInfoClass() # reset for the next iteration

            if time.time() - lastSaveTime > 3600 :
                lastSaveTime = time.time()
                saveCheckPoint(savedCheckPoint+"_hourly.pth",
                               epoch-1, imer, minGEpoch, minGdLoss/normRec,
                               generator, discriminator,
                               optimizer_G, optimizer_D,
                               startFrom=it, interimRes=resAcc)
                saveModels(f"model_{TCfg.exec}_hourly")


        resAcc *= 1.0/totalIm
        writer.add_scalars("Losses per epoch",
                           {'Dis': resAcc.lossD
                           ,'Adv': resAcc.lossGA
                           ,'Gen': lossAdvCoef * resAcc.lossGA + lossDifCoef * resAcc.lossGD
                           }, epoch )
        writer.add_scalars("Distances per epoch",
                           {'MSE': resAcc.lossMSE
                           ,'L1L': resAcc.lossL1L
                           ,'REC': resAcc.lossGD
                           }, epoch )
        writer.add_scalars("Probs per epoch",
                           {'Ref': resAcc.predReal
                           ,'Gen': resAcc.predFake
                           ,'Pre': resAcc.predPre
                           }, epoch )
        lastGdLossTrain = resAcc.lossGD

        Rec_test, MSE_test, L1L_test, Rprob_test, Fprob_test, Dloss_test, GAloss_test, GDloss_test \
            = summarizeSet(testLoader, False)
        writer.add_scalars("Test per epoch",
                           {'MSE': MSE_test / normTestMSE
                           ,'L1L': L1L_test / normTestL1L
                           ,'REC': Rec_test / normTestRec
                           #,'Dis': Dis_test
                           #,'Gen': Gen_test
                           }, epoch )
        writer.add_scalars("Test losses per epoch",
                           { 'Dis': Dloss_test
                           , 'Adv': GAloss_test
                           , 'Gen': lossAdvCoef * GAloss_test + lossDifCoef * GDloss_test
                           }, epoch )
        writer.add_scalars("Test probs per epoch",
                           {'Ref': Rprob_test
                           ,'Gen': Fprob_test
                           }, epoch )

        lastGdLoss = Rec_test
        if lastGdLoss < minGdLoss  :
            minGdLoss = lastGdLoss
            minGEpoch = epoch
            saveCheckPoint(savedCheckPoint+"_B.pth",
                           epoch, imer, minGEpoch, minGdLoss,
                           generator, discriminator,
                           optimizer_G, optimizer_D)
            os.system(f"cp {savedCheckPoint}.pth {savedCheckPoint}_BB.pth") # BB: before best
            os.system(f"cp {savedCheckPoint}_B.pth {savedCheckPoint}.pth") # B: best
            saveModels(f"model_{TCfg.exec}_B")
        else :
            saveCheckPoint(savedCheckPoint+".pth",
                           epoch, imer, minGEpoch, minGdLoss,
                           generator, discriminator,
                           optimizer_G, optimizer_D)
        saveModels()

        resAcc = TrainResClass()
        afterEachEpoch(epoch)


def testMe(tSet, imags=1):
    if isinstance(imags, int) :
        testSubSet = [ tSet.__getitem__() for _ in range(imags) ]
    else :
        testSubSet = [ tSet.__getitem__(index) for index in imags ]
    testImages = torch.stack( [ testItem[0] for testItem in testSubSet ] ).to(TCfg.device)
    colImgs, probs, dists = generateDiffImages(testImages, layout=4)
    testIndeces = []
    for im in range(len(testSubSet)) :
        testItem = testSubSet[im]
        testIndeces.append(testItem[1])
        print(f"Index: ({testItem[1]})")
        print(f"Probabilities. Org: {probs[im,0]:.3e},  Gen: {probs[im,2]:.3e},  Pre: {probs[im,1]:.3e}.")
        print(f"Distances. Rec: {dists[im,0]:.4e},  MSE: {dists[im,1]:.4e},  L1L: {dists[im,2]:.4e}.")
        plotImage(colImgs[im].squeeze().cpu())
    return testIndeces




def freeGPUmem() :
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()






