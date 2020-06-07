import cv2
import numpy as np
import os
import random
from sklearn.utils import shuffle

def nothing(image, angle):
    '''this is function that does nothing'''
    return image, angle

def flip(image, angle):
    '''flip the image'''
    image = cv2.flip(image, 1)
    return image, -angle

def brightness(image, angle):
    '''adjust the brightness in hsv channel and convert back to rgb'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:,:,2] =  cv2.add(hsv[:,:,2],random.randrange(-50, 50))
    image =  cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image, angle

def smooth(image, angle):
    '''apply a smoothing filter'''
    image = cv2.GaussianBlur(image,(3,3), 0)
    return image, angle

# augmentation options, the augmentation can be one of the four.
preprocess_options = {
    0 : nothing,
    1 : flip,
    2 : brightness,
    3 : smooth}

def augment_list(samples, ratio = 0.5):
    """
    generating augment list
    """
    # First combining left, center, right images. For left and right images, adjust the steering angle.
    new_samples = []
    angle_offset = 0.2
    for sample in samples:
        # format is [path, steering angle, preprocess option].
        angle = float(sample[3])
        new_samples.append([sample[0], angle, 0]) # center
        new_samples.append([sample[1], angle + angle_offset, 0]) # left
        new_samples.append([sample[2], angle - angle_offset, 0]) # right
    
    shuffle(new_samples)
    augment_size = int(len(new_samples) * ratio)
    
    # Randomly generating augmentation options. Preprocessing is not done here, it is done in the generator
    # in runtime. Here, only the options are recorded.
    augmented_samples = []
    for i in range(augment_size):
        sample = new_samples[i]
        sample[2] = random.randrange(1, 4)
        augmented_samples.append(sample)
        
    return new_samples + augmented_samples

def generator(samples, batch_size = 32):
    '''generate images at the runtime'''
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../track2data/IMG/' + batch_sample[0].split('\\')[-1]
                image = cv2.imread(name)
                angle = batch_sample[1]
                
                image, angle = preprocess_options[batch_sample[2]](image, angle)
                # crop the image
                image = image[50:-20,:]
                image = cv2.resize(image, (200, 66))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                
                images.append(image)
                angles.append(angle)
                
            x_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(x_train, y_train)