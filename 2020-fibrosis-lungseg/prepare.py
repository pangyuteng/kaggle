import os, sys
import json
from tqdm import tqdm

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from scipy import ndimage
from skimage import measure
import SimpleITK as sitk

import matplotlib.pyplot as plt

def seed_everything(seed=2020):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
seed_everything(42)


#https://gist.github.com/pangyuteng/fdbf0e13cd9173dc11aabccb30f8a2ad

import albumentations as A

aug_pipeline = A.Compose([
    A.ShiftScaleRotate(p=0.5),    
])

def readimage(image_file,mask_file,augment):

    _x_sample = np.load(image_file).astype(np.float)
    x_sample = np.copy(_x_sample)
    
    y_mask = np.load(mask_file)
    y_mask = y_mask.astype(np.float)
    
    minval,maxval = -1000,1000
    x_sample = (x_sample-minval)/(maxval-minval)
    x_sample = x_sample.clip(0,1)
        
    
    augmented = aug_pipeline(
        image=x_sample,
        mask=y_mask,
    )
    x_sample = augmented['image']
    y_sample = augmented['mask']
    

    x_sample = np.expand_dims(x_sample,axis=-1)
    y_sample = np.expand_dims(y_sample,axis=-1)

    return x_sample, y_sample

# https://github.com/keras-team/keras/issues/9707

from tensorflow.keras.utils import Sequence
class MyDataGenerator(Sequence):
    def __init__(self,data_list,batch_size=8,shuffle=True,augment=False):
        
        self.x = np.array([x['img'] for x in data_list])
        self.y = np.array([x['mask'] for x in data_list])

        self.indices = np.arange(len(self.y))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
    def __len__(self):
        return int(np.floor(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.shuffle:
            inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_x = self.x[inds]
            batch_y = self.y[inds]
        else:
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            
        # read your data here using the batch lists, batch_x and batch_y
        x = []
        y = []
        for image_file,mask_file, in zip(batch_x,batch_y):
            ix,iy=readimage(image_file,mask_file,self.augment)
            x.append(ix)
            y.append(iy)

        x,y = np.asarray(x, dtype=np.float), np.asarray(y, dtype=np.float)
        #print(type(x),type(y),x.shape,y.shape)
        return x,y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# https://gist.github.com/pangyuteng/7f54dbfcd67fb9d43a85f8c6818fca7b

import os
import SimpleITK as sitk
import pydicom
import traceback
from scipy import ndimage

def get_slice_location(dcm_file):
    ds = pydicom.dcmread(dcm_file)
    try:
        val = float(ds.SliceLocation)
    except:
        val = float('nan')
        
    return dcm_file, val

def imread(fpath):

    if os.path.isdir(fpath):
        
        file_list = os.listdir(fpath)
        file_loc_list = sorted([get_slice_location(os.path.join(fpath,x)) for x in file_list],key=lambda x:x[1])    
        dicom_names = [x[0] for x in file_loc_list]

        # check if intercept and slope is your typical 
        dcm = pydicom.dcmread(dicom_names[-1])
        b = dcm.RescaleIntercept
        m = dcm.RescaleSlope
        if (b,m) != (-1024, 1):
            raise ValueError(f'bad ct dicom! {fpath}')


        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(dicom_names)
        img = reader.Execute()

    else:

        reader= sitk.ImageFileReader()
        reader.SetFileName(fpath)
        img = reader.Execute()

    arr = sitk.GetArrayFromImage(img)        
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()    
    return arr,spacing,origin,direction

def imwrite(fpath,arr,spacing,origin,direction,use_compression=True):
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection(direction)
    writer = sitk.ImageFileWriter()    
    writer.SetFileName(fpath)
    writer.SetUseCompression(use_compression)
    writer.Execute(img)
    
def _read_img(folder,patient_id):
    fpath = os.path.join(folder,patient_id)
    arr,spacing,origin,direction = imread(fpath)
    arr = arr.astype(np.int16)
    return arr,spacing,origin,direction


# naive lung with old school image processing technique!
def get_lung_mask(arr):
    
    bkgd = np.zeros(arr.shape).astype(np.uint8)
    bkgd[:,:,:2]=1
    bkgd[:,:,-2:]=1
    bkgd[:,:2,:]=1
    bkgd[:,-2:,:]=1
    
    # assume < 0 HU are voxels within lung
    procarr = (arr < -300).astype(np.int)
    procarr = ndimage.morphology.binary_closing(procarr,iterations=1)

    label_image, num = ndimage.label(procarr)
    region = measure.regionprops(label_image)

    region = sorted(region,key=lambda x:x.area,reverse=True)
    lung_mask = np.zeros(arr.shape).astype(np.uint8)
    
    # assume `x` largest air pockets except covering bkgd is lung, increase x for lung with fibrosis (?)
    x=3
    for r in region[:x]: # should just be 1 or 2, but getting x, since closing may not work.
        mask = label_image==r.label
        contain_bkgd = np.sum(mask*bkgd) > 0
        if contain_bkgd > 0:
            continue
        lung_mask[mask==1]=1
    
    lung_mask = ndimage.morphology.binary_dilation(lung_mask,iterations=1)
    
    return lung_mask


root = '/kaggle/input/osic-pulmonary-fibrosis-progression'
temp_dir = '/kaggle/temp'
raw_list_path = os.path.join(temp_dir,'raw_list.json')

train_csv = os.path.join(root,'train.csv')
train_folder = os.path.join(root,'train')

test_csv = os.path.join(root,'test.csv')
test_folder = os.path.join(root,'test')
if __name__ == '__main__':


    img_root = os.path.join(temp_dir,'image_nii')
    mask_root = os.path.join(temp_dir,'mask_nii')
    raw_img_root = os.path.join(temp_dir,'slice','img')
    raw_mask_root = os.path.join(temp_dir,'slice','mask')

    os.makedirs(img_root,exist_ok=True)
    os.makedirs(mask_root,exist_ok=True)
    os.makedirs(raw_img_root,exist_ok=True)
    os.makedirs(raw_mask_root,exist_ok=True)

    # perform lung seg.
    train_ids = sorted(os.listdir(train_folder))
    my_folder = train_folder

    lung_list = train_ids[-2:]

    # create image and mask nifti file.
    for patient in tqdm(lung_list):
        
        img_path = os.path.join(img_root,f'{patient}.nii.gz')
        lung_mask_path = os.path.join(mask_root,f'{patient}.nii.gz')
        
        if os.path.exists(img_path) and os.path.exists(lung_mask_path):
            continue

        img,spacing,origin,direction = _read_img(my_folder,patient)
        mask = get_lung_mask(img)
            
        mask = mask.astype(np.uint8)
        imwrite(lung_mask_path,mask,spacing,origin,direction)    
        imwrite(img_path,img,spacing,origin,direction)
    
    # for sake of simplicity save as individual slices
    if os.path.exists(raw_list_path):
        print('skip')
    else:
        raw_list = []

        print(len(lung_list))
        for patient in tqdm(lung_list):

            uid = patient
            axial_img_path = os.path.join(raw_img_root,f'img_{uid}_0.npy')
            axial_mask_path = os.path.join(raw_mask_root,f'mask_{uid}_0.npy')

            tmp=dict(
                img=axial_img_path,
                mask=axial_mask_path,
            )
            if tmp in raw_list:
                continue
            
            img_path = os.path.join(img_root,f'{patient}.nii.gz')
            lung_mask_path = os.path.join(mask_root,f'{patient}.nii.gz')

            img,spacing,origin,direction = imread(img_path)
            mask,spacing,origin,direction = imread(lung_mask_path)
            
            for x in range(img.shape[0]):
                try:
                    axial_img_path = os.path.join(raw_img_root,f'img_{uid}_{x}.npy')
                    axial_mask_path = os.path.join(raw_mask_root,f'mask_{uid}_{x}.npy')
                    axial_img = img[x,:,:].squeeze().astype(np.int16)
                    axial_mask = mask[x,:,:].squeeze().astype(np.uint8)
                    np.save(axial_img_path,axial_img)
                    np.save(axial_mask_path,axial_mask)
                    
                    raw_list.append(dict(
                        img=axial_img_path,
                        mask=axial_mask_path,
                    ))
                except:
                    pass
                
        with open(raw_list_path,'w') as f:
            f.write(json.dumps(raw_list))

    with open(raw_list_path,'r') as f:
        raw_list = json.loads(f.read())

    print(len(raw_list))
    batch_size = 15
    dg = MyDataGenerator(raw_list,batch_size=batch_size)
    print(len(dg))
    myx,myy = dg[4]
    print(myx.shape,myy.shape)
    
    for x in range(batch_size):
        plt.subplot(211)
        plt.imshow(myx[x,:].squeeze(),cmap='gray')
        plt.subplot(212)
        plt.imshow(myy[x,:].squeeze(),cmap='gray')
        plt.savefig(f'x_viz_prepare{x}.png')
        plt.close()

    print(np.max(myx),np.min(myx))
    print(np.max(myy),np.min(myy))
    