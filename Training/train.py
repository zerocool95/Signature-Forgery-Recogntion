import os
import cv2
import keras
import numpy as np
import random
from tqdm import tqdm 

from model.model import CustomModel

class Trainer():
    def __init__(self, batch_size = 3, epochs =30, no_of_training_triplets =30000, no_of_testing_triplets = 10000):
        self.batch_size = batch_size
        self.epochs = epochs
        self.no_of_training_triplets = no_of_training_triplets
        self.no_of_testing_triplets = no_of_testing_triplets

    def make_triplet_datastructure(self):
        print(os.getcwd())
        if self.batch_size % 3 != 0 :
            raise "Invalid batch size"
        with open("data/triplet_test_seen.txt","r") as f:
            triplet_test_data = f.read().split("\n")
        for i in range(len(triplet_test_data)):
            triplet_test_data[i] = triplet_test_data[i].split(",")
            try: 
                for j in range(len(triplet_test_data[i])):
                    triplet_test_data[i][j] = triplet_test_data[i][j].split("/")[2]
            except:
                print(i,triplet_test_data[i][j])
        k = len(triplet_test_data) % self.batch_size
        triplet_test_data = triplet_test_data[:-k]
        with open("data/triplet_train.txt","r") as f:
            triplet_train_data = f.read().split("\n")
        for i in range(len(triplet_train_data)):
            triplet_train_data[i] = triplet_train_data[i].split(",")
            try: 
                for j in range(len(triplet_train_data[i])):
                    triplet_train_data[i][j] = triplet_train_data[i][j].split("/")[2]
            except:
                print(i,triplet_train_data[i][j])
        k = len(triplet_train_data) % self.batch_size
        triplet_train_data = triplet_train_data[:-k]

        return triplet_test_data, triplet_train_data

    def get_images_dict(self):
        images =  os.listdir("data/images/")
        image_dic = {}
        for i in tqdm(images, total = len(images)):
            image_dic[i] = cv2.imread("data/images/"+i,0)/255.

        return image_dic


    def train_gen(self, train_list, image_dic):
        mask = np.zeros((150,275))
        while True:
            random.shuffle(train_list)
            trimmed_train_list = train_list[:self.no_of_training_triplets]
            yield_list = []
            for i,j in enumerate(trimmed_train_list):
                if i!=0 and i % (self.batch_size/3)==0:
                    out = np.array(yield_list).reshape(self.batch_size,150,550, 1)
                    
                    #out = np.stack((out,out,out), axis = 0)
                    # np.stack((out,out,out), axis = 3).reshape(batch_size,150,550, 3),
                    yield [out, out, out], np.zeros((self.batch_size, 1))
                    yield_list = []
                mask_index = 0#random.randint(0,3)
                img1 = np.copy(image_dic[self.triplet_train_data[j][0]])
                img2 = np.copy(image_dic[self.triplet_train_data[j][1]])
                img3 = np.copy(image_dic[self.triplet_train_data[j][2]])
                if mask_index == 1:
                    img1[:,:275] = mask
                    img2[:,:275] = mask
                    img3[:,:275] = mask
                elif mask_index == 2:
                    img1[:,275:] = mask
                    img2[:,275:] = mask
                    img3[:,275:] = mask
                yield_list.append(img1)
                yield_list.append(img2)
                yield_list.append(img3)
    
    def test_gen(self, test_list, image_dic):
        while True:
            random.shuffle(test_list)
            yield_list = []
            trimmed_test_list = test_list[:self.no_of_testing_triplets]
            for i,j in enumerate(trimmed_test_list):
                if i!=0 and i % (self.batch_size/3)==0:
                    out = np.array(yield_list).reshape(self.batch_size,150,550, 1)
                    #out = np.stack((out,out,out), axis = 0)
                    #np.stack((out,out,out), axis = 3).reshape(batch_size,150,550, 3),
                    yield [out, out, out], np.zeros((self.batch_size, 1))
                    yield_list = []
                yield_list.append(image_dic[self.triplet_test_data[j][0]])
                yield_list.append(image_dic[self.triplet_test_data[j][1]])
                yield_list.append(image_dic[self.triplet_test_data[j][2]])

    def train(self):
        self.triplet_test_data, self.triplet_train_data = self.make_triplet_datastructure()
        print(len(self.triplet_test_data), len(self.triplet_train_data))

        train_steps = (len(self.triplet_train_data)*3)/self.batch_size
        test_steps = (len(self.triplet_test_data)*3)/self.batch_size

        train_list = []
        for k in range(len(self.triplet_train_data)):
            train_list.append(k)

        test_list = []
        for k in range(len(self.triplet_test_data)):
            test_list.append(k)

        image_dic = self.get_images_dict()

        train_generator = self.train_gen(train_list, image_dic)
        test_generator = self.train_gen(test_list, image_dic)


        if os.path.exists('data/weights/') == False:
            os.mkdir('data/weights/')

        custom_model_obj = CustomModel()
        triplet_model  = custom_model_obj.siamese_model()

        checkpoint = keras.callbacks.ModelCheckpoint('weights/weights.{epoch:02d}.hdf5',
                                        monitor='val_loss', 
                                        verbose=0, 
                                        save_best_only=False, 
                                        save_weights_only=True, 
                                        mode='auto', period=1)


        triplet_model.fit_generator(train_generator,
                        steps_per_epoch = train_steps,
                        epochs = self.epochs ,
                        callbacks = [checkpoint],
                        validation_data = test_generator,
                        validation_steps = test_steps,
                        )



        return

