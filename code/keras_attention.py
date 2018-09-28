#Self Attention Layer
from keras import backend as K
from keras.engine.topology import Layer
import numpy as np

class NonMasking(Layer):   
    def __init__(self, **kwargs):   
        self.supports_masking = True  
        super(NonMasking, self).__init__(**kwargs)   
  
    def build(self, input_shape):   
        input_shape = input_shape   
  
    def compute_mask(self, input, input_mask=None):   
        # do not pass the mask to the next layers   
        return None   
  
    def call(self, x, mask=None):   
        return x   
  
    def compute_output_shape(self, input_shape):   
        return input_shape  

class Attention(Layer):

    def __init__(self, time_step, output_dim, **kwargs):
        self.output_dim = output_dim
        self.time_step = time_step
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[-1],self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        #print(self.kernel.shape)
        super(Attention, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        a = K.reshape(K.softmax(K.sum(K.dot(x, self.kernel),axis=-1)), (-1,self.time_step,1))
        #a = K.softmax(K.sum(K.dot(x, self.kernel),axis=-1))
        return a

    def compute_output_shape(self, input_shape):
        b = (input_shape[0], input_shape[1])
        #print(b)
        return b
    
class AttentionWrapper(Layer):

    def __init__(self, **kwargs):
        super(AttentionWrapper, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape
        self.input_shape_ = input_shape


    def call(self, x):
        self.hidden_state = x[0]
        self.weight = x[1]
        print(self.hidden_state.shape)
        print(self.weight.shape)
        
        '''This is wrong'''
        #h = K.reshape(self.hidden_state, (15,1024))
        #w = K.reshape(self.weight, (20,1)) 
        #print('w', w.shape)
        #mul = h * w#[-1][0]
        
        mul = K.sum(self.hidden_state * self.weight, axis=1)
        #print('mul', mul.shape)
        return mul

    def compute_output_shape(self, input_shape):
        b = (input_shape[0][0], int(self.hidden_state.shape[-1]))
        #print(b)
        return b
