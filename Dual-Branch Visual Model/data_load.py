from torch.utils.data import Dataset
import os
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
import numpy
import scipy.io as sio


def transform(x, y):
    if numpy.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y
def getLabel(vid_index, frame_index):
    fixdatafile = ('/home/tianh/envv2.7/CDNN-traffic-saliency-master/fixdata/fixdata' + str(vid_index) + '.mat')
    data = sio.loadmat(fixdatafile)

    fix_x = data['fixdata'][frame_index - 1][0][:, 3]
    fix_y = data['fixdata'][frame_index - 1][0][:, 2]
    fix_x = fix_x.astype('int')
    fix_y = fix_y.astype('int')
    mask = numpy.zeros((720, 1280), dtype='float32')
    #print(len(fix_x),vid_index, frame_index)
    for i in range(len(fix_x)):
        #print(fix_x[i],fix_y[i])
        mask[fix_x[i], fix_y[i]] = 1
    mask = filters.gaussian_filter(mask, 40)
    mask = numpy.array(mask, dtype='float32')
    mask = cv2.resize(mask, (320,192), interpolation=cv2.INTER_CUBIC)
    mask = mask.astype('float32') / 255.0
    if mask.max() == 0:
        print (mask.max())
        #print img_name
    else:
        mask = mask / mask.max()
    return mask

class ImageList(Dataset):
    def __init__(self, root, imgs, for_train=False):
        self.root = root
        self.imgs = imgs
        self.for_train = for_train

    def __getitem__(self, index):
        img_name = self.imgs[index]
        # vid_index = int(img_name[0:2])
        # frame_index = int(img_name[3:9])
        # image_name = os.path.join(self.root, img_name)
        # img = io.imread(image_name)
        # img = cv2.resize(img, (320,192), interpolation=cv2.INTER_CUBIC)
        # img = img.astype('float32')/255.0
        
        imgarr=[]
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])
        for m in range (6):
            fra_index=frame_index-m
            #print(fra_index)
            vid_index=vid_index
            img_name='%02d'%(vid_index) +"/" +'%06d'%(fra_index)  +'.jpg'
            #print(img_name)
            image_name = os.path.join(self.root, img_name)
            img = io.imread(image_name)
            img = cv2.resize(img, (320,192), interpolation=cv2.INTER_CUBIC)
            img = img.transpose(2, 0, 1)
            img = numpy.ascontiguousarray(img)
            #print(type(img))
            imgarr.append(torch.from_numpy(img))
        imgarr=torch.stack(imgarr)
        imgarr = imgarr.float()/255.0
        mask = getLabel(vid_index, frame_index)
        # if self.for_train:
            # img, mask = transform(img, mask)
        mask = mask[None, ...]
        mask = numpy.ascontiguousarray(mask)
        #exit(0)
        return imgarr, torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)

