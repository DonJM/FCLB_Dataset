#import neccessary python modules and libraries
import os
import shutil
from PIL import Image
import uuid
import tarfile
import imgaug as ia
import imageio
import imgaug.augmenters as iaa
import random
import numpy as np

class FCLB_Datasets:
    #class' constructor
    def __init__(self, setup):
        self.setup = setup

    #I call it segment function, wala na koy mahunahunaan
    def segment(self):
        original_dir = self.setup.getWorkspaces()['ORIGINAL_IMAGES_PATH']
        target_dir = self.setup.getWorkspaces()['PREPROCESSED_IMAGE_PATH']

        original_path = os.path.join(original_dir)
        target_path = os.path.join(target_dir)

        for label in self.setup.labels:
            originalFiles = os.path.join(original_path, label)
            targetFilePath = os.path.join(target_path, label)
            for img in os.listdir(originalFiles):
                self.segmentation(originalFiles, targetFilePath, img, "musa acuminata-{}".format(label))
 

    #this function used to segment an image file, it actually resize image, add contrast and lightness
    #and save it to the target directory
    #the purpose of uuid module is to give unique id or name to an image file
    def segmentation(self, base_dir, target_dir, fname, label):
        image = Image.open(os.path.join(base_dir, fname))
        size = (640, 640)
        image.thumbnail(size)

        #initiate variable n to add random number ranging from 0 to 360
        n = random.randint(0,360)
        image = np.asarray(image)
        
        #rotating images
        rotate=iaa.Affine(rotate=(-(n), n))
        rotated_image=rotate.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), rotated_image)
        
        #adding noise
        gaussian_noise=iaa.AdditiveGaussianNoise(10,20)
        noise_image=gaussian_noise.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), noise_image)
        
        #image cropping
        crop = iaa.Crop(percent=(0, 0.3)) # crop image
        crop_image=crop.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), crop_image)
        
        #images padding and shearing
        shear = iaa.Affine(shear=(0,40))
        shear_image=shear.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), shear_image)
        
        
        #flipping image horizontally
        flip_hr=iaa.Fliplr(p=1.0)
        flip_hr_image= flip_hr.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), flip_hr_image)
        
        #flippine images vertically
        flip_vr=iaa.Flipud(p=1.0)
        flip_vr_image= flip_vr.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), flip_vr_image)
        
        #adding contrast to the image
        contrast=iaa.GammaContrast(gamma=0.4)
        contrast_image =contrast.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), contrast_image)
        
        #adding contrast to the image
        contrast=iaa.GammaContrast(gamma=0.9)
        contrast_image =contrast.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), contrast_image)
        
        #images scalling
        scale_im=iaa.Affine(scale={"x": (1.5, 1.0), "y": (1.5, 1.0)})
        scale_image =scale_im.augment_image(image)
        imageio.imsave(os.path.join(target_dir, label + "-"+ '{}.jpg'.format(uuid.uuid1())), scale_image)