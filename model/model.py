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
from keras.optimizers import SGD , Adam, RMSprop

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

class CustomModel():
    def __init__(self, width = 550, height = 150, channel = 1, final_embedding_size = 2048, batch_size = 3):
        self.input_width = width
        self.input_height = height
        self.input_channel = channel
        self.final_embedding_size = 2048
        self.batch_size = batch_size

    def MultiHeadsAttModel(self,l=8*8, d=512, dv=64, dout=512, nv = 8 ):

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

    
    def network_1(self):
        first_input = Input(shape=(self.input_height,self.input_width,self.input_channel))
        x = Conv2D(8, (5, 5), activation='relu', padding='same')(first_input)
        x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(5, 5), strides=(4, 4))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same',)(x)
        x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same',)(x)
        x = MaxPooling2D(pool_size=(2, 4), strides=(2, 4))(x)
        first_max = Flatten()(x)
        first_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(first_max)

        return first_max, first_input

    def network_2(self):
        second_input = Input(shape=(self.input_height,self.input_width,self.input_channel))
        second_conv = Conv2D(96, kernel_size=(8, 8),strides=(8,8), padding='same')(second_input)
        second_max = MaxPool2D(pool_size=(7,7),strides = (2,2),padding='same')(second_conv)
        second_conv = Conv2D(96, kernel_size=(8, 8),strides=(8,8), padding='same')(second_max)
        second_max = MaxPool2D(pool_size=(1,3),padding='same')(second_conv)
        second_max = Flatten()(second_max)
        second_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(second_max)

        return second_max, second_input

    def network_3(self):
        third_input = Input(shape = (self.input_height,self.input_width,self.input_channel))
        third_max = Conv2D(32,(3,3),activation='relu', padding='same')(third_input)
        third_max = MaxPooling2D(pool_size=(3, 3))(third_max)
        third_max = Conv2D(64,(3,3),activation='relu')(third_max)
        third_max = MaxPooling2D(pool_size=(4, 4), padding='same')(third_max)
        third_max = Conv2D(64*3,(5,5),activation='relu')(third_max)
        third_max = MaxPooling2D(pool_size=(2, 4), padding='same')(third_max)
        third_max = Reshape([44,192])(third_max)    
        att = self.MultiHeadsAttModel(l=44, d=192 , dv=8*3, dout=32, nv = 8 )
        third_max = att([third_max,third_max,third_max])
        third_max = Reshape([4,11,32])(third_max)   
        third_max = NormL()(third_max)
        third_max = Flatten()(third_max) 
        third_max = Lambda(lambda  x: K.l2_normalize(x,axis=1))(third_max)

        return third_max, third_input


    def _loss_tensor(self,y_true,y_pred):
        loss=tf.convert_to_tensor(0,dtype=tf.float32)
        total_loss=tf.convert_to_tensor(0,dtype=tf.float32)
        g=tf.constant(1.0,shape=[1],dtype=tf.float32)
        zero=tf.constant(0.0,shape=[1],dtype=tf.float32)
        for i in range(0,self.batch_size,3):
            try:
                q_embedding=y_pred[i]
                p_embedding=y_pred[i+1]
                n_embedding=y_pred[i+2]
                D_q_p=K.sqrt(K.sum((q_embedding-p_embedding)**2))
                D_q_n=K.sqrt(K.sum((q_embedding-n_embedding)**2))
                loss=tf.maximum(g+D_q_p-D_q_n,zero)
                total_loss=total_loss+loss
            except:
                continue
        total_loss=total_loss/(self.batch_size/3)
        return total_loss

    def siamese_model(self):
        out_1, inp_1  = self.network_1()
        out_2, inp_2 = self.network_2()
        out_3, inp_3 = self.network_3()

        merge_one = concatenate([out_1, out_2])

        emb = concatenate([merge_one, out_3])
        emb = Dropout(0.01)(emb)
        emb = Dense(2048)(emb)

        final_model = Model(inputs=[inp_1, inp_2, inp_3], outputs=emb)

        final_model.compile(loss=self._loss_tensor, optimizer= RMSprop(lr = 1e-4)) 

        return final_model