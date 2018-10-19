import torch

import numpy as np
import json
from sklearn.neighbors import KNeighborsClassifier
import time

from utils import *
import multiprocessing

# load trained embedding
# load testing embedding

begin_time = time.time()
train_embedding = torch.load("embedding_39.pth").detach().numpy()
with open("embedding_meta_39.json", "r") as f:
	train_embedding_meta = json.load(f)
train_labels = np.array(train_embedding_meta["label"])
train_image_paths = np.array(train_embedding_meta["image_path"])

test_embedding = torch.load("embedding_test.pth").detach().numpy()
with open("embedding_meta_testing.json", "r") as f:
	test_embedding_meta = json.load(f)
test_labels = np.array(test_embedding_meta["label"])
train_image_paths = np.array(test_embedding_meta["image_path"])

print("training shape")
print(train_embedding.shape)
print(train_labels.shape)


print("testing shape")
print(test_embedding.shape)
print(test_labels.shape)
print("fiting knn")
fit_time = time.time()
knn = KNeighborsClassifier(n_neighbors=30, n_jobs=-1)
knn.fit(train_embedding, train_labels)
print("Fit time: %s" %(format_time(time.time()-fit_time)))
train_embedding = None
print("preicting")
total_accuracy = 0.
batch_size = 10000
# for i in range(1000):
pred_time = time.time()


# p = multiprocessing.Process(target=knn.kneighbors, args=(test_embedding[:batch_size].reshape(batch_size,-1)))
pred = knn.kneighbors(test_embedding[:batch_size].reshape(batch_size,-1))
# jobs = []
# jobs.append(p)
# p.start()
# for proc in jobs:
	# proc.join()
print("Pred time: %s" %(format_time(time.time()-pred_time)))

# print(pred)
# print(pred[1].shape)
# print(test_labels[:batch_size])
pred_labels = train_labels[pred[1]]
print(pred_labels.shape)
for i in range(batch_size):
	# print(pred_labels[i] == test_labels[i])
	# print((pred_labels[i] == test_labels[i]).shape)
	sample_acc = np.sum(pred_labels[i]==test_labels[i])/30
	total_accuracy += sample_acc
	#print(total_accuracy/(i+1)*100)
	progress_bar(i, batch_size, "Acc %.2f" %((total_accuracy/(i+1)*100)))
# print(total_accuracy/10*100)


print("Total time: %s" %(format_time(time.time()-begin_time)))

