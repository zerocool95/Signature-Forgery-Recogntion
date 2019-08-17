
from common.preprocessing import Preprocessing
import sys
import cv2
import numpy as np
from model.model import CustomModel
from common.utils import Utils
from scipy.spatial import distance

class Predictor():
    def __init__(self):
        self.preprocessing = Preprocessing()
        self.utils = Utils()

    def predict(self, img1, img2):
        f1 = cv2.imread(img1.strip())

        if f1 is None:
            print("Image "+img1+" not found.")  
            sys.exit()
        f1 = self.preprocessing.process(f1)
        f2 = cv2.imread(img2.strip())
        if f2 is None:
            print("Image "+img2+" not found.") 
            sys.exit()
        f2 = self.preprocessing.process(f2)

        self.utils.show_images_sidebyside(f1, f2)
        f1 = np.array([f1.reshape(150,550,1)])
        f2 = np.array([f2.reshape(150,550,1)])
        

        custom_model_obj = CustomModel()
        pred_model  = custom_model_obj.siamese_model()
        pred_model.load_weights('data/weights/best/weights.hdf5')

        emb_f1 = pred_model.predict([f1,f1,f1])
        emb_f2 = pred_model.predict([f2,f2,f2])
        
        dist = distance.euclidean(emb_f1[0],emb_f2[0])

        thresh = 2.33

        distance_between_images = dist#get_predicted_dist(img1_path,img2_path)
        print("----------------------------------------------------")
        print("Distance between images : ", distance_between_images)
        print("----------------------------------------------------")
        if distance_between_images >= thresh:
            print("Forgery Detected.")
        else:
            print("Both Images are from the same writer.")