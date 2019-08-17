import os
import cv2

class Trainer():
	def __init__(self, batch_size):
		self.batch_size = batch_size

	def make_triplet_datastructure(self):
		if self.batch_size % 3 !=0 :
		    raise "Invalid batch size"
		with open("triplet_with_forgery_test.txt","r") as f:
		    triplet_test_data = f.read().split("\n")
		for i in range(len(triplet_test_data)):
		    triplet_test_data[i] = triplet_test_data[i].split(",")
		    try: 
		        for j in range(len(triplet_test_data[i])):
		            triplet_test_data[i][j] = triplet_test_data[i][j].split("/")[1]
		    except:
		        print(i,triplet_test_data[i][j])
		k = len(triplet_test_data) % self.batch_size
		triplet_test_data = triplet_test_data[:-k]
		with open("triplet_with_forgery_train.txt","r") as f:
		    triplet_train_data = f.read().split("\n")
		for i in range(len(triplet_train_data)):
		    triplet_train_data[i] = triplet_train_data[i].split(",")
		    try: 
		        for j in range(len(triplet_train_data[i])):
		            triplet_train_data[i][j] = triplet_train_data[i][j].split("/")[1]
		    except:
		        print(i,triplet_train_data[i][j])
		k = len(triplet_train_data) % self.batch_size
		triplet_train_data = triplet_train_data[:-k]

		return triplet_test_data, triplet_train_data

	def train(self):
		pass

