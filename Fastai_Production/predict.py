from fastai import *
from fastai.vision import *
import io
import pickle
import cloudpickle

def predict(inputs):
    import io
    from fastai.vision import open_image
    from fastai import basic_train


    model = load_learner('./')
    for img in inputs:
        img_tensor = open_image(io.BytesIO(img))
    pred = [str(model.predict(img_tensor)[0])]

    return pred

    
