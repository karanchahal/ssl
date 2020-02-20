from models.simclr import Simclr
from models.mnist import MnistNet
from models.classifier import Classifier
models = {
    'simclr': Simclr,
    'mnist' : MnistNet,
    'classifier' : Classifier,
}

def get_model(name):
    return models[name]