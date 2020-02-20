from models.simclr import Simclr
from models.mnist import MnistNet

models = {
    'simclr': Simclr,
    'mnist' : MnistNet,
}

def get_model(name):
    return models[name]