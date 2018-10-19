import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io
import random
import pprint
from PIL import Image
from pylab import imread
import json

class TinyImageNetDataset(Dataset):
	def __init__(self, root, train=True, transform=None, debug=False):
		self.debug = debug
		self.debug_limit = 2 if debug else 200
		# self.debug_classes = []

		self.train = train
		self.root = root
		self.files = os.path.join(root, "train") if train else os.path.join(root, "val")
		self.transform = transform
		self.cache = None
		self.classes_labels = None
		
		self.class_folders = os.listdir(self.files)   # array of all the classes id
		self.classes_labels = {}					  # a dictionary maps from class id to its true English label
		
		self.images = []
		self.labels = []

		self.counter = 0
		
		self._set_up()

	def __len__(self):
		'''
		returns the length of the dataset
		'''
		return len(self.labels)

	def __getitem__(self, idx):
		'''
		load the image and transform
		'''
		
		#print(len(self.labels))
		#print(self.counter)
		

		sample = self.images[idx]
		label = self.labels[idx]
		ret_dict = {}
		if not self.train:
			image = Image.open(sample)
			
			ret_dict["image"] = self._convert_to_rgb(image)
			ret_dict["label"] = label
			ret_dict["image_path"] = sample
			# print(image.size)
			# print(np.array(image).shape)
			if self.transform:
				ret_dict["image"] = self.transform(ret_dict["image"])
			
		else:
			

			for key in sample.keys():
				temp = sample[key]
				ret_dict[key] = Image.open(sample[key])
				ret_dict[key] = self._convert_to_rgb(ret_dict[key])
				# print(sample[key].size)
				# print(np.array(sample[key]).shape)
				# sample[key].show()
				if self.transform:
					ret_dict[key] = self.transform(ret_dict[key])

			ret_dict["image_path"] = temp
			ret_dict["label"] = label
			
		self.counter += 1
		if self.train and self.counter >= len(self.labels):
			torch.cuda.synchronize()
			self.counter = 0
			self._construct_images_list()
			print("Epoch ended, reshuffle random images")
		return ret_dict

	def _convert_to_rgb(self, image):
		"""
		converts the image to rgb
		"""
		rgbimg = Image.new("RGB", image.size)
		rgbimg.paste(image)
		return rgbimg


	def _set_up(self):
		self._get_classes_labels()
		self._construct_images_list()
		print(len(self.images))
		

	def _construct_images_list(self):
		'''
		set up the images and labels 
		for training images, it's an array of dictionary {image, positive, negative}
		all the entries are the filename path

		:params
		:return
		'''
		if not self.train:
			# if in validation, return a list of images files names for reading
			with open(os.path.join(self.files, "val_annotations.txt")) as f:
				for line in f:
					info = line.split("\t") # file name, label 
					# print(info[1])
					# if (info[1] == "n03763968") or (info[1] == "n04540053"):
					# if info[1] != 'n03763968' or info[1] != "n04540053":
					# 	# continue
					image_path = os.path.join(self.files, "images", info[0])
					self.images.append(image_path)
					self.labels.append(info[1])
				assert(len(self.labels) == len(self.images))
				return 
		

		classes = list(self.classes_labels.keys())
		assert(len(classes) == 200)

		all_images_filepath = {}
		for folder in self.class_folders:
			if len(list(all_images_filepath.keys())) >= self.debug_limit:
				break
			all_images_filepath[folder] = []
			for image_file in os.listdir(os.path.join(self.files, folder, "images")):
				image_file_path = os.path.join(self.files, folder, "images", image_file)
				all_images_filepath[folder].append(image_file_path)
		print(len(all_images_filepath))
		assert(len(all_images_filepath) == self.debug_limit)
		assert(len(all_images_filepath[self.class_folders[0]]) == 500)

		self._setup_triplets(all_images_filepath)

	def _get_classes_labels(self):
		'''
		get the label name of each class id
		"wnids.txt" contains the 200 id 
		"words.txt" contains the id, label in single line
			e.g.: "id \t id_label,..."
		:params
		:return
		'''

		id_dict_path = os.path.join(self.root, "wnids.txt")
		id_dict_file = open(id_dict_path, "r+")
		id_dict = {}
		for line in id_dict_file:
			line = line.split("\n")[0]
			id_dict[line] = None
			
		# loads the file
		id_to_label_dict_path = os.path.join(self.root, "words.txt")
		id_to_label_dict_file = open(id_to_label_dict_path, "r+")
		for line in id_to_label_dict_file:
			id = line.split("\t")[0]
			label = line.split("\t")[1].split("\n")[0]
			if id in id_dict.keys():
				self.classes_labels[id] = label
		# note can dump this to json
		id_to_label_dict_file.close()
		id_dict_file.close()
		print("Finished getting the classes labels")
	

	def _setup_triplets(self, all_images_filepath):
		'''
		set up the triplet
		:params
			all_images_filepath: dictioanry of label to image file paths
		:return
			triplet: a dictionary of query_image label, positive label, negative label
		'''
		for i, label in enumerate(list(all_images_filepath.keys())):
			for j, image_file in enumerate(all_images_filepath[label]):
				random_postive_offset = random.randint(1, len(all_images_filepath[label])-1) # 500
				random_postive_index = (j+random_postive_offset) % len(all_images_filepath[label])
				positive_file = all_images_filepath[label][random_postive_index]

				random_negative_folder_offset = random.randint(1, len(all_images_filepath)-1) # 
				random_negative_folder_index = (i+random_negative_folder_offset) % (len(all_images_filepath)) # 200
				random_negative_label = list(all_images_filepath.keys())[random_negative_folder_index]
				random_negative_image_index = random.randint(0, len(all_images_filepath[random_negative_label])-1) # 500
				negative_file = all_images_filepath[random_negative_label][random_negative_image_index]
				
				self.images.append({"image": image_file, "positive": positive_file, "negative": negative_file})
				self.labels.append(label)
				assert(positive_file!=image_file)
				assert(random_negative_label!=label)

def main():
	dataset = TinyImageNetDataset("../data/tiny-imagenet-200/", train=True, transform=
																			transforms.Compose([
																				transforms.ToTensor(),
																			]))
	# class_labels = dataset.classes_labels
	# print(class_labels)
	# print(len(dataset))
	# # name= Image.open(os.path.join(dataset.files, "n03976657", "images", "n03976657_431.JPEG")
	# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
	# # name.show()
	# # plt.imshow(name)
	# for i, sample_batch in enumerate(dataloader):
	# 	print(i)
	# 	print(sample_batch['image'].shape)
	# 	# print(i, sample_batch['image'].shape, sample_batch['positive'].shape, class_labels[sample_batch['label']], sample_batch['image_path'])
	# 	image = transforms.ToPILImage()(sample_batch['image'][0])

	# 	image.show()
		
	# 	if i == 3:
	# 		break

if __name__ == "__main__":
	main()
