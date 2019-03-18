# Panini Guide

Panini is a platform that serves ML/DL models at low latency and makes the ML model deployment to production from a few days to a few minutes. Once deployed in Panini’s server, it will provide you with an API key to infer the model. Panini query engine is developed in C++, which provides very low latency during model inference and Kubernetes cluster is being used to store the model so, it is scalable to multiple nodes. Panini also takes care of caching and batching inputs during model inference. We support most frameworks in Python and we have tested so far for PyTorch, SciKit, Tensorflow, Keras, and Spark. 

![alt text](https://panini.ai/static/img/Panini_deployment_reduce.png)

## What to Upload?

To get started, you need to upload three files.

- requirements.txt
- main.py
- Your saved model file. It could be named anything and the valid extensions are .pkl or .pth
- (Optional) You can also upload additional python .py files or helper files such as .csv .txt 

## What is main.py? 

main.py is the first python file that gets executed in the container. It needs to have a function called             predict inside of it which gets executed when you call your API to make predictions. It also takes a single argument which is the data send for prediction. Forward pass of the data will be performed here and the return value from predict function will be send back to the user. 

predict(input_from_client):

Arguments:
- input_from_client. This is a list and you need to access each individaul element before doing forward pass. 

Returns:
- Predicted value which has to be a double array of list.

Ie. 
return_value = my_model(input_from_client)
   
return [[return_value]]

#### Template for main.py
```python
#main.py

# 1. imports. Make sure sklearn and numpy is in requirements.txt file
from sklearn import linear_model
from numpy import array

def predict(input_from_client):

    #2. Load my saved model
    model = load("model.pkl")

    #3. Do the prediction
    prediction = model.predict(input_from_client)
    value = []
    for label in prediction:
        if label == 0:
            value.append('Setosa')
        elif label == 1:
            value.append('Virginica')
        else:
            value.append('Versicolour')
    
    # 4. Return the predicted value back to the user.
    return value

#You can have more helper methods and classes if you want. 
def blah(...):
  pass

def blah2(...):
  pass



```



## What is requirements.txt ?
“Requirements files” are files containing a list of items to be installed using pip install. 
Look into this link for more info. https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format


## Specifying model input type?

When uploading your model, you have to specify input type the model is expecting. If your model is an image classifier and is expecting an image, you will use "bytes" since the image with be encoded as base64 bytes

- ints: Model expecting integer as input
- double: Model expecting double as input
- floats: Model expecting floats as input
- bytes: Model expecting bytes as input. Ie. Image classification models
- strings: Model expecting strings as input. Ie. text classification and NLP models

## How to use API to make predictions? 

As long as you can make a POST request, you can use panini. We are platfrom agnostic when it comes to infering your model.

Here is a snippet on using Python to make prediction. 

 
```python
import json
import requests
import base64
API_LINK = "" #Your API URL goes here
data_to_send = [] #Data you want to send for prediction goes here
response = requests.post(
     API_LINK,
     headers={"Content-type": "application/json"},
     data=json.dumps({
         'input': data_to_send,
     }))
result = response.json()
print(result) #Prediction response
```





1. SciKit-Learn_Production
  - Tutorial Video: https://youtu.be/KdoP1k_M6h4
  - Medium Post: https://medium.com/@avinregmi/deploy-scikit-learn-into-production-under-2-minutes-cae873b25f9b


 2. PyTorch_CNN_Production (Deploy CNN model to Production via Panini)

 - Tutorial Video: https://youtu.be/tCz-fi_NheE
 - Medium Post: https://medium.com/@avinregmi/deploy-ml-dl-models-to-production-via-panini-3e0a6e9ef14

