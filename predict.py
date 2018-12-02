import os
import shutil
import numpy as np
import cv2
from scipy.spatial import distance
from model import seamese_model
from preprocessing import process
import sys
if(len(sys.argv) != 3):
    print("Use of command : python3 predict.py signature1.jpg signature2.jpg")
    sys.exit()
img1_path = sys.argv[1]
img2_path = sys.argv[2]
pred_model = seamese_model()
pred_model.load_weights('./weights.hdf5')
def get_predicted_dist(img1, img2):
#     print(img1, img2)
    
    f1 = cv2.imread(img1.strip())
    if f1 is None:
        print("Image "+img1+" not found.")  
        sys.exit()
    f1 = process(f1)
    f2 = cv2.imread(img2.strip())
    if f2 is None:
        print("Image "+img2+" not found.") 
        sys.exit()
    f2 = process(f2)
    f1 = np.array([f1.reshape(150,550,1)])
    f2 = np.array([f2.reshape(150,550,1)])
    emb_f1 = pred_model.predict([f1,f1,f1])
    emb_f2 = pred_model.predict([f2,f2,f2])
    
    dist = distance.euclidean(emb_f1[0],emb_f2[0])
    
    return dist
thresh = 2.33
distance_between_images = get_predicted_dist(img1_path,img2_path)
print("----------------------------------------------------")
print("Distance between images : ", distance_between_images)
print("----------------------------------------------------")
if distance_between_images >= thresh:
    print("Forgery Detected.")
else:
    print("Both Images are from the same writer.")
print("----------------------------------------------------")
