# 1. import everything needed 
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL.Image
import io



def load_my_model():
    # 2. defie model architecture
    model = models.densenet121(pretrained=True)

    #Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))
    
    # 3. load weights
    model.classifier.load_state_dict(torch.load("last_layers.pth",map_location='cpu'))
    
    # 4. return weights loaded model
    return model
    

    
def predict(input_image_client):
    # 2. define preprocess for the input
    preprocess = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # 3. input from user will come as a list, even if there is only one input. Loop through the input and apply 
    #transofrm to each input and convert into tensor.
    for input_img in input_image_client:
        img_tensor = preprocess(PIL.Image.open(io.BytesIO(input_img)))[:3] #Just use 3 channels
        img_tensor = img_tensor.view(1,3,224,-1)

    return_label = ""
    model = load_my_model()
    # 4.  We perform a forward pass image tensor into our model
    with torch.no_grad():
        model.eval()
        model.cpu()
        logps = model(img_tensor)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        if top_class == 0:
            return_label = "Cat Prob:{}".format(top_p)
        else:
            return_label = "Dog Prob:{}".format(top_p)

    # 5. length of return variable must be equal to final output from model. In our case
    # very last layer in our model is nn.Linear(256, 2). So, it is expecting list with length of 2.
    # I'm just adding dummy value of 1 to make list length of two.
    
    return_label = [return_label,1]
    
    # 5. return the prediction back to the user
    return return_label