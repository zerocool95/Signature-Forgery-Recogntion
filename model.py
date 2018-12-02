from keras.callbacks import ModelCheckpoint
from keras.layers import *
from keras.models import Model, load_model
from keras.optimizers import SGD , Adam
from keras.callbacks import BaseLogger, TensorBoard, CSVLogger
from keras.preprocessing.image import load_img, img_to_array
from skimage import transform
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras import objectives
from keras import backend as K

def MultiHeadsAttModel(l=8*8, d=512, dv=64, dout=512, nv = 8 ):

    v1 = Input(shape = (l, d))
    q1 = Input(shape = (l, d))
    k1 = Input(shape = (l, d))

    v2 = Dense(dv*nv, activation = "relu")(v1)
    q2 = Dense(dv*nv, activation = "relu")(q1)
    k2 = Dense(dv*nv, activation = "relu")(k1)

    v = Reshape([l, nv, dv])(v2)
    q = Reshape([l, nv, dv])(q2)
    k = Reshape([l, nv, dv])(k2)
        
    att = Lambda(lambda x: K.batch_dot(x[0],x[1] ,axes=[-1,-1]) / np.sqrt(dv),
                 output_shape=(l, nv, nv))([q,k])# l, nv, nv
    att = Lambda(lambda x:  K.softmax(x) , output_shape=(l, nv, nv))(att)

    out = Lambda(lambda x: K.batch_dot(x[0], x[1],axes=[4,3]),  output_shape=(l, nv, dv))([att, v])
    out = Reshape([l, d])(out)
    
    out = Add()([out, q1])

    out = Dense(dout, activation = "relu")(out)

    return  Model(inputs=[q1,k1,v1], outputs=out)

class NormL(Layer):

    def __init__(self, **kwargs):
        super(NormL, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.a = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='ones',
                                      trainable=True)
        self.b = self.add_weight(name='kernel', 
                                      shape=(1,input_shape[-1]),
                                      initializer='zeros',
                                      trainable=True)
        super(NormL, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        eps = 0.000001
        mu = K.mean(x, keepdims=True, axis=-1)
        sigma = K.std(x, keepdims=True, axis=-1)
        ln_out = (x - mu) / (sigma + eps)
        return ln_out*self.a + self.b

    def compute_output_shape(self, input_shape):
        return input_shape

def seamese_model():
 
   # base_model = VGG16(include_top = False, input_shape = (150,550,3))  
   # first_max = Conv2D(512, (4, 4),strides = (4,4), activation='relu', padding='same')(base_model.output)
   # print("first_max : ",first_max.shape)
   # first_max = Flatten()(first_max)
    
   # mymodel = Model(inputs = base_model.input, outputs = first_max)
    
   # for layer in base_model.layers[:-1]:
   #     layer.trainable = False
#     first_input = Input(shape=(150,550,1))
#     first_conv = Conv2D(96, kernel_size=(8, 8),strides=(16,16), padding='same')(first_input)
#     first_max = MaxPool2D(pool_size=(3,3),strides = (4,4),padding='same')(first_conv)
#     first_max = Flatten()(first_max)
#     first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

    first_input = Input(shape=(150,550,1))
#     x = Conv2D(64, (3, 3), input_shape=(150,550,1), padding='same',
#            activation='relu')(first_input)
    x = Conv2D(8, (5, 5), activation='relu', padding='same')(first_input)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
#     x = Conv2D(128, (3, 3), activation='relu', padding='same',)(x)
#     x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
#     x = Conv2D(96, (3, 3), activation='relu', padding='same',)(x)
#     x = Conv2D(256, (3, 3), activation='relu', padding='same',)(x)
#     x = Conv2D(256, (3, 3), activation='relu', padding='same',)(x)
    x = MaxPooling2D(pool_size=(5, 5), strides=(4, 4))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',)(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same',)(x)
#     x = Conv2D(512, (3, 3), activation='relu', padding='same',)(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
#    x = Conv2D(512, (3, 3), activation='relu', padding='same',)(x)
#   x = Conv2D(512, (3, 3), activation='relu', padding='same',)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',)(x)
    x = MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(x)
    #x = Conv2D(128, (4, 4),strides = (4,4), activation='relu', padding='same',)(x)
    #print('first x: ', x.shape)
    first_max = Flatten()(x)
    first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)
    
    second_input = Input(shape=(150,550,1))
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(8,8), padding='same')(second_input)
    second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
    second_conv = Conv2D(96, kernel_size=(8, 8),strides=(8,8), padding='same')(second_max)
    
    second_max = MaxPool2D(pool_size=(1,3),padding='same')(second_conv)
    #print('second max: ', second_max.shape)
    second_max = Flatten()(second_max)
    second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)
    
    third_input = Input(shape = (150,550,1))
    third_max = Conv2D(32,(3,3),activation='relu', padding='same')(third_input)
    third_max = MaxPooling2D(pool_size=(3, 3))(third_max)
    third_max = Conv2D(64,(3,3),activation='relu')(third_max)
    third_max = MaxPooling2D(pool_size=(4, 4), padding='same')(third_max)
    third_max = Conv2D(64*3,(5,5),activation='relu')(third_max)
    third_max = MaxPooling2D(pool_size=(2, 4), padding='same')(third_max)
    #print(third_max.shape)
    #print([third_max.shape[1]*third_max.shape[2],third_max.shape[3]])
    third_max = Reshape([44,192])(third_max)    
    att = MultiHeadsAttModel(l=44, d=192 , dv=8*3, dout=32, nv = 8 )
    third_max = att([third_max,third_max,third_max])
    #print(third_max.shape)
    third_max = Reshape([4,11,32])(third_max)   
    third_max = NormL()(third_max)
    third_max = Flatten()(third_max) 
    third_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(third_max)

    merge_one = concatenate([first_max, second_max])

    emb = concatenate([merge_one, third_max])
    emb = Dropout(0.01)(emb)
    emb = Dense(2048)(emb)
    #l2_norm_final = Lambda(lambda  x: K.l2_normalize(x,axis=1))(emb)

    final_model = Model(inputs=[first_input, second_input, third_input], outputs=emb)

    return final_model