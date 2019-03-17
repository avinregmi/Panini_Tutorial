from sklearn import linear_model
from numpy import array
def load(path):
    import pickle
    infile = open(path,'rb')
    model = pickle.load(infile)
    infile.close()
    return model


def predict(input_from_client):
    model = load("model.pkl")
    prediction = model.predict(input_from_client)
    value = []
    for label in prediction:
        if label == 0:
            value.append('Setosa')
        elif label == 1:
            value.append('Virginica')
        else:
            value.append('Versicolour')
    
    return value