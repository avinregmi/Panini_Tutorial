# Panini Guide

Panini is a platform that serves ML/DL models at low latency and makes the ML model deployment to production from a few days to a few minutes. Once deployed in Panini’s server, it will provide you with an API key to infer the model. Panini query engine is developed in C++, which provides very low latency during model inference and Kubernetes cluster is being used to store the model so, it is scalable to multiple nodes. Panini also takes care of caching and batching inputs during model inference. We support most frameworks in Python and we have tested so far for PyTorch, SciKit, Tensorflow, Keras, and Spark. 

![alt text](https://panini.ai/static/img/Panini_deployment_reduce.png)

## What to Upload?

To get started, you need to upload three files.

- requirements.txt
- main.py
- Your saved model file. It could be named anything and the valied extensions are .plk or .pth
- (Optional) You can also upload addiotioanl python .py classes or helper files such as .csv .txt files. 

## What is main.py? 

main.py is the first python file that gets executed in the container. It needs to have a function called             predict inside the main.py which gets executed when you call your API to make predictions. It also takes a single argument which is the data send for prediction. Forward pass of the data will be performed here and the return value from predict function will be send back to the user. 

predict(input_from_client):

Arguments:
- input_from_client. This is a list and you need to access each individaul element before doing forward pass. 

Returns:
- Predicted value which has to be a double array of list.

Ie. return_value = my_model(input_from_client)
    return [[return_value]]


## What is requirements.txt ?
“Requirements files” are files containing a list of items to be installed using pip install. 
Look into this link for more info. https://pip.pypa.io/en/stable/reference/pip_install/#requirements-file-format


## Specifying model input type?


## How to use API to make predictions? 



1. SciKit-Learn_Production
  - Tutorial Video: https://youtu.be/KdoP1k_M6h4
  - Medium Post: https://medium.com/@avinregmi/deploy-scikit-learn-into-production-under-2-minutes-cae873b25f9b


 2. PyTorch_CNN_Production (Deploy CNN model to Production via Panini)

 - Tutorial Video: https://youtu.be/tCz-fi_NheE
 - Medium Post: https://medium.com/@avinregmi/deploy-ml-dl-models-to-production-via-panini-3e0a6e9ef14

