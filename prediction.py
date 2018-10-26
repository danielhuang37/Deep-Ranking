import torch

import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
import time

from utils import *
import multiprocessing
import codecs


def main(mode="test"):

	begin_time = time.time()
	
	
	############### loads the training and testing embedding #####################
	
	train_embedding = torch.load("./src/embedding_15.pth").detach().numpy()
	with open("./src/embedding_meta_15.json", "r") as f:
		train_embedding_meta = json.load(f)
	train_labels = np.array(train_embedding_meta["label"])
	train_image_paths = np.array(train_embedding_meta["image_path"])

	test_embedding = torch.load("./embedding_"+mode+".pth").detach().numpy()
	with open("./embedding_meta_"+mode+".json", "r") as f:
		test_embedding_meta = json.load(f)
	test_labels = np.array(test_embedding_meta["label"])
	test_image_paths = np.array(test_embedding_meta["image_path"])


	
	################ Fiting knn ########################
	
	print("fiting knn")
	fit_time = time.time()
	knn = KNeighborsClassifier(n_neighbors=30, n_jobs=6)
	knn.fit(train_embedding, train_labels)
	print("Fit time: %s" %(format_time(time.time()-fit_time)))
	print("preicting")

	total_accuracy = 0.
	batch_size = 10000 if mode == "test" else 5

	############# Predicting knn ######################

	pred_time = time.time()
	pred = knn.kneighbors(test_embedding[:batch_size].reshape(batch_size,-1))
	print("Pred time: %s" %(format_time(time.time()-pred_time)))


	pred_labels = train_labels[pred[1]]
	pred_image_paths = train_image_paths[pred[1]]
	pred_distances = pred[0]

	demo_save_dict = {}
	for i in range(batch_size):
		if mode=="demo":
			demo_save_dict[test_labels[i]] = {
			"image_path": test_image_paths[i],
			"top_labels": pred_labels[i][:10].tolist(),
			"top_distance": pred_distances[i][:10].tolist(),
			"top_image_paths": pred_image_paths[i][:10].tolist(),

			"bot_labels": pred_labels[i][-10:].tolist(),
			"bot_distance": pred_distances[i][-10:].tolist(),
			"bot_image_paths": pred_image_paths[i][-10:].tolist()
			}
		else:
		
			sample_acc = np.sum(pred_labels[i] == train_labels[i])/30
			
			total_accuracy += sample_acc
		
	print(total_accuracy/(i)*100)


	print("Total time: %s" %(format_time(time.time()-begin_time)))
	with open("demo_info.json", "w") as f:
		json.dump(demo_save_dict, f, indent=4)

if __name__ == "__main__":
	main(mode="demo")
