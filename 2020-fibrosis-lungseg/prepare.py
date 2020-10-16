import os, sys
import traceback
from tqdm import tqdm
import json
import copy

import random
import numpy as np
import pandas as pd
import tensorflow as tf

import imageio
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
    x_sample = np.load(image_file).astype(np.float)
    y_mask = np.load(mask_file).astype(np.float)
    
    minval,maxval = -1000,1000
    x_sample = (x_sample-minval)/(maxval-minval)
    x_sample = x_sample.clip(0,1)

    if not augment:
        x_sample = np.expand_dims(x_sample,axis=-1)
        y_sample = np.expand_dims(y_mask,axis=-1)
        return x_sample, y_sample

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
        
        self.x = np.array([x['npy_img'] for x in data_list])
        self.y = np.array([x['npy_mask'] for x in data_list])

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
        ds = pydicom.dcmread(dicom_names[-1])
        b = ds.RescaleIntercept
        m = ds.RescaleSlope
        if (b,m) != (-1024, 1):
            raise ValueError(f'bad rescale slope! {fpath}')

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

    # scaled
    png_img_root = os.path.join(temp_dir,'png','img')
    png_mask_root = os.path.join(temp_dir,'png','mask')

    # unscaled
    npy_img_root = os.path.join(temp_dir,'npy','img')
    npy_mask_root = os.path.join(temp_dir,'npy','mask')

    os.makedirs(img_root,exist_ok=True)
    os.makedirs(mask_root,exist_ok=True)

    os.makedirs(png_img_root,exist_ok=True)
    os.makedirs(png_mask_root,exist_ok=True)

    os.makedirs(npy_img_root,exist_ok=True)
    os.makedirs(npy_mask_root,exist_ok=True)

    
    train_ids = sorted(os.listdir(train_folder))
    my_folder = train_folder

    lung_list = []
    for patient in train_ids:
        fpath = os.path.join(my_folder,patient)
        count = len(os.listdir(fpath))
        if 50 < count < 300: # demo purpose, trim down list
            lung_list.append(patient)

    print('lung_list len',len(lung_list))
    tmp_lung_list = copy.copy(lung_list)
    for patient in tqdm(tmp_lung_list):
        try:
            fpath = os.path.join(my_folder,patient)
            dicom_names = [os.path.join(fpath,x) for x in os.listdir(fpath)]
            # check if intercept and slope is your typical 
            ds = pydicom.dcmread(dicom_names[-1])
            b = ds.RescaleIntercept
            m = ds.RescaleSlope

            if (b,m) != (-1024, 1): # didn't want to bother with those with non standard ct slopes.
                raise ValueError(f'bad rescale slope! {fpath}')
            if ds.pixel_array.shape != (512,512): # lung seg only works with 512, FOV for >=768 is odd.
                raise ValueError(f'bad shape! {fpath}')

        except:
            traceback.print_exc()
            lung_list.remove(patient)

    print('lung_list len',len(lung_list))
    lung_list = lung_list[-10:] # demo purpose just use 10 series

    # create image and mask nifti file.
    for patient in tqdm(lung_list):
        
        img_path = os.path.join(img_root,f'{patient}.nii.gz')
        lung_mask_path = os.path.join(mask_root,f'{patient}.nii.gz')
        
        if os.path.exists(img_path) and os.path.exists(lung_mask_path):
            continue

        img,spacing,origin,direction = _read_img(my_folder,patient)
        # perform naive lung seg.
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

            img_path = os.path.join(img_root,f'{patient}.nii.gz')
            lung_mask_path = os.path.join(mask_root,f'{patient}.nii.gz')

            img,spacing,origin,direction = imread(img_path)
            img = img.astype(np.int16) # typical ct data type
            
            mask,spacing,origin,direction = imread(lung_mask_path)
            mask = mask.astype(np.uint8)

            # scale image from -1000,1000 to 0,255 (optional)
            img_uint8 = ((img+1000)/2000).clip(0,1)*255
            img_uint8 = img_uint8.astype(np.uint8)
            mask_uint8 = (mask*255).clip(0,255).astype(np.uint8)

            for x in range(img.shape[0]):
                try:

                    # NPY
                    npy_axial_img = img[x,:,:].squeeze()
                    npy_axial_mask = mask[x,:,:].squeeze()
                    npy_axial_img_path = os.path.join(npy_img_root,f'img_{patient}_{x}.npy')
                    npy_axial_mask_path = os.path.join(npy_mask_root,f'mask_{patient}_{x}.npy')
                    np.save(npy_axial_img_path,npy_axial_img)
                    np.save(npy_axial_mask_path,npy_axial_mask)

                    # PNG
                    png_axial_img = img_uint8[x,:,:].squeeze()
                    png_axial_mask = mask_uint8[x,:,:].squeeze()
                    png_axial_img_path = os.path.join(png_img_root,f'img_{patient}_{x}.png')
                    png_axial_mask_path = os.path.join(png_mask_root,f'mask_{patient}_{x}.png')
                    imageio.imwrite(png_axial_img_path,png_axial_img)
                    imageio.imwrite(png_axial_mask_path,png_axial_mask)

                    raw_list.append(dict(
                        npy_img=npy_axial_img_path,
                        npy_mask=npy_axial_mask_path,
                        png_img=png_axial_img_path,
                        png_mask=png_axial_mask_path,
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
    myx,myy = dg[2]
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
    