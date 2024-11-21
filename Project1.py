import numpy as np

#import torch which has many of the functions to build deep learning models and to train them
import torch
import torch.nn as nn
import torch.optim as optim

#import torchvision, which was lots of functions for loading and working with image data
import torchvision
import torchvision.transforms as transforms

#this is a nice progress bar representation that will be good to measure progress during training
import tqdm

class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128,10)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        y = self.relu(self.fc2(x))
        return y

class MyClassifier():
    
    ''' Do not change the class name. Do not change any of the existing function names. You may add extra functions as you see fit.'''
    
    def __init__(self):
        self.class_labels = ['edible_1', 'edible_2', 'edible_3', 'edible_4', 'edible_5',
                            'poisonous_1', 'poisonous_2', 'poisonous_3', 'poisonous_4', 'poisonous_5']
        
        
    def setup(self):
        ''' This function will initialise your model. 
            You will need to load the model architecture and load any saved weights file your model relies on.
        '''
        dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.dino_model=dino
        self.dino_model.eval()
        
        self.model=LinearClassifier()
        self.model.load_state_dict(torch.load('first_model.pth', map_location='cpu'))
        self.model.eval()

        imagenet_means = (0.485, 0.456, 0.406)
        imagenet_stds = (0.229, 0.224, 0.225)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(imagenet_means, imagenet_stds)])
        
    def test_image(self, image):
        ''' This function will be given a PIL image, and should return the predicted class label for that image. 
            Currently the function is returning a random label.  
        '''
        transformed_im=self.transform(image).unsqueeze(0)
        dino_out=self.dino_model(transformed_im)
        output=self.model(dino_out)

        predicted_idx=torch.argmax(output)
              
        predicted_cls = self.class_labels[predicted_idx]
        return predicted_cls