import os
import imp
import torch
from mmdnn.conversion.pytorch.pytorch_parser import PytorchParser
from mmdnn.conversion.caffe.caffe_emitter import CaffeEmitter
from mmdnn.conversion.caffe.saver import save_model

class Pytorch2Caffe:
    def __init__(self, model, save_root, save_name, input_shape):
        self.parse = PytorchParser(model, input_shape)
        self.save_root = save_root
        # self.save_root = os.path.dirname(os.path.realpath(__file__)) + '/save/mmdnn/'
        self.save = {
            'structurejson': os.path.join(self.save_root, save_name, '.json'),
            'structurepb': os.path.join(self.save_root, save_name, '.pb'),
            'weights': os.path.join(self.save_root, save_name, '.npy'),

            'caffenetwork': os.path.join(self.save_root, save_name, '.py'),
            'caffeweights': os.path.join(self.save_root, save_name, '.cnpy'),
            'caffemodel': os.path.join(self.save_root, save_name),
            'caffeproto': os.path.join(self.save_root, save_name)
        }

    def start(self):
        print("start to do pytorch to IR")
        self.parse.run(self.save_root)
        print("done! then to do IR to caffe code")
        emitter = CaffeEmitter((self.save['structurepb'], self.save['weights']))
        emitter.run(self.save['caffenetwork'], self.save['caffeweights'], 'test')
        print("done! then to do ccode to model")
        MainModel = imp.load_source('MainModel', self.save['caffenetwork'])
        save_model(MainModel, self.save['caffenetwork'], self.save['caffeweights', 
                                        self.save['caffemodel'], self.save['caffeproto']])
        print("done!!!!!!")
        

