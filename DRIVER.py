# Import dependencies

import cv2
import sys
import torch
import scipy as sp
from os.path import isfile
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam
from tqdm import tqdm
import random
from apex import amp
from efficientnet_pytorch import EfficientNet

torch.cuda.empty_cache()

# Requires SampleSubmission to be in the same directory.

def crop_black_border(img,tol = 2,debug=False):
    try:
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(gray_img,0,255,cv2.THRESH_OTSU)
        bbox= cv2.boundingRect(thresh)
        x,y,w,h= bbox
        crop = img[y:y+h,x:x+w]
        return crop
    except:
        raise Exception('Image is None')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Constants

seed_everything(1234)
TTA = 10
num_classes = 1
lr = 1e-3
IMG_SIZE = 256
NEW_RADIUS = 500
test = sys.argv[1] 
output_name = sys.argv[2]

# Weights to load to create ensemble

b5_weights = 'b5v_weights.pt'
# Coefficients for b5_weights = [0.85802471,1.46767681,2.20268473,2.97144944]
b5_weights_k = 'b5k_weights.pt'
# Coefficients for b5_weights_k = [0.91005243,1.40109634,2.16976472,2.97504493]
b4_weights = 'b4v_weights.pt'
# Coefficients for b4_weights = [0.90798377,1.47056216,2.10613042,3.06465985]

def expand_path(p):
    p = str(p)
    if isfile(test+p):
        return test + p 
    return p

# Inspired by Benjamin Graham's MinPooling preprocessing (Diabetic Retinopathy Detection winner)

def calculate_scale_factor(centre_array,NEW_SIZE):
    centre_array = centre_array.sum(1)
    radius = (centre_array>centre_array.mean()/10).sum()/2
    try:
        s = NEW_SIZE*1.0/radius
    except:
        return -1
    return s

class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None, DUMMY_COUNTER = 0):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.Expected.values[idx]
        label = np.expand_dims(label, -1)
        p = self.df.Id.values[idx]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        centre_array = (image[int(image.shape[0]/2),:,:])
        scale_factor = calculate_scale_factor(centre_array,NEW_RADIUS)
        resized_image = cv2.resize(image,(0,0),fx=scale_factor,fy=scale_factor)
        resized_image = cv2.addWeighted(resized_image,4,
                                        cv2.GaussianBlur(resized_image,(0,0),NEW_RADIUS/30), -4,
                                        128)
        mask_circle = np.zeros(resized_image.shape)
        cv2.circle(mask_circle, (int(resized_image.shape[1]/2),int(resized_image.shape[0]/2)),int(NEW_RADIUS*0.90),(1,1,1),-1)
        resized_image = resized_image * mask_circle +  (1-mask_circle)
        resized_image = resized_image.astype('uint8')
        resized_image = crop_black_border(resized_image)
        resized_image = cv2.resize(resized_image,(IMG_SIZE,IMG_SIZE))
        image = transforms.ToPILImage()(resized_image)

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations for Test Time Augmentation - Constant for all models

test_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-360, 360)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

testset = MyDataset(pd.read_csv('SampleSubmission.csv'),
                 transform=test_transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

# EfficientNet-b5 loading

# Load model. 
model = EfficientNet.from_pretrained('efficientnet-b5')
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.cuda()
optimizer = Adam(model.parameters(),lr = lr,weight_decay = 1e-5)
criteria = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 5, gamma = 0.1)
model, optimizer = amp.initialize(model,optimizer,opt_level = "O1")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Prediction utility script

def predict(X, coef):
    X_p = np.copy(X)
    for i, pred in enumerate(X_p):
        if pred < coef[0]:
            X_p[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            X_p[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            X_p[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            X_p[i] = 3
        else:
            X_p[i] = 4
    return X_p

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Load weights.
model.load_state_dict(torch.load(b5_weights))

# Prediction Setup 
for param in model.parameters():
    param.requires_grad = False

sample = pd.read_csv('SampleSubmission.csv')
test_pred = np.zeros((len(sample), 1))

# Test Time Augmentation
model.eval()

for _ in range(TTA):
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            images, _ = data
            images = images.cuda()
            pred = model(images)
            test_pred[i * 16:(i + 1) * 16] += pred.detach().cpu().squeeze().numpy().reshape(-1, 1)

output = test_pred / TTA
coefficient_test_output = output

# Fit for all validation predictions and all targets
# Optimize coefficients on validation data - Coefficients calculated to optimize quadratic weighted kappa on a 10% validation set.

optimized_coefficients = [0.85802471,1.46767681,2.20268473,2.97144944]

test_predictions = predict(coefficient_test_output, optimized_coefficients)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# Creating CSV
submission = pd.DataFrame({'Id':pd.read_csv('SampleSubmission.csv').Id.values,
                          'Expected':np.squeeze(test_predictions).astype(int)})

print(submission.head())

submission.to_csv(output_name, index=False)
