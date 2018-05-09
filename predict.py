import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from PIL import Image

import json

# Implement the code to predict the class from an image file
def predict(image_path, model, topk=5, labels_json=''):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    was_training = model.training    
    model.eval()
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(image_path)
    pil_image = img_loader(pil_image).float()
    
    image = np.array(pil_image)    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (np.transpose(image, (1, 2, 0)) - mean)/std    
    image = np.transpose(image, (2, 0, 1))
    
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG
    
    if torch.cuda.is_available():
        image = image.cuda()
            
    result = model(image).topk(topk)

    probs = result[0].data.numpy()[0]
    classes = result[1].data.numpy()[0]
    if labels_json:
        with open('cat_to_name.json', 'r') as f:
            cat_to_name = json.load(f)

        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
        
    model.train(mode=was_training)
        
    return probs, classes
