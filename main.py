import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import torchvision
from torchvision import transforms, models
import torch.optim as optim
import sys
import time
import argparse
from utils import *
from dataset import * 
import json
import sklearn.neighbors as neighbors
from torchsummary import summary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--pretrain', type=int, default='0', help="pretrain model or not")
parser.add_argument('--lr', type=float, default=0.001, help="leanring rate")
parser.add_argument('--bs', type=int, default=4, help="batch size")
opt = parser.parse_args()

size = 224

train_transform_config = [
	# transforms.RandomAffine(45),
	transforms.RandomHorizontalFlip(),
	#transforms.RandomRotation(20),
	# transforms.RandomCrop(size, padding=4),  
	transforms.ToTensor(), 
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]

train_transform_config = [transforms.Resize((size, size))] + train_transform_config
train_transform = transforms.Compose(train_transform_config)


test_transform_config = [
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	]

test_transform_config = [transforms.Resize((size, size))] + test_transform_config
test_transform = transforms.Compose(test_transform_config)


train_set = TinyImageNetDataset("../data/tiny-imagenet-200/", \
								train=True, \
								transform=train_transform, debug=False)
train_loader = torch.utils.data.DataLoader(train_set,\
												batch_size=opt.bs,\
												shuffle=True,\
												num_workers=4)

test_set = TinyImageNetDataset("../data/tiny-imagenet-200/", \
								train=False, \
								transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set,\
											batch_size=opt.bs,\
											shuffle=False,\
											num_workers=8) 

loss_tracker = LossTracker()

def adjust_learning_rate(optimizer, epoch, k=0.01):
	lr = opt.lr * np.exp(-k*epoch)
	print("Learning rate: %f" % (lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def train(net, criterion, optimizer, epoch, init_e=0):
	e = init_e
	
	while(e < epoch):
		net.train()
		
		embedding_saver = []
		embedding_meta = {"label": [], "image_path": []}

		e += 1
		running_loss = 0.
		print(len(train_loader))
		
		adjust_learning_rate(optimizer, e-1)


		for i, data in enumerate(train_loader):
			

			query_image, positive_image, negative_image, label = data['image'], data['positive'], data['negative'], data['label']
			# print(query_image.shape)
			query_image, positive_image, negative_image = query_image.to(device), positive_image.to(device), negative_image.to(device)

			optimizer.zero_grad()

			query_output = net(query_image)
			positive_output = net(positive_image)
			negative_output = net(negative_image)
			# print(query_output)

			
			loss = criterion(query_output, positive_output, negative_output)
			loss.backward()

			optimizer.step()

			running_loss += loss.item()

			if e%3==0:
				embedding_saver.append(query_output.cpu())
				embedding_meta["label"] += data['label']
				embedding_meta["image_path"] += data["image_path"]

			
			progress_bar(i, len(train_loader), "Loss: %f"%((running_loss/(i+1))))
			
			
		print(" ")
		epoch_loss = running_loss/(i+1)
		loss_tracker.append(epoch_loss)
		print("Epoch: %d | Loss: %.3f" % (e, epoch_loss))
		if e%3 == 0:
			torch.save(net.state_dict(), "checkpoint_"+str(e)+".pt")
			embedding = torch.cat(embedding_saver)
			torch.save(embedding, "embedding_"+str(e)+".pth")
			
			with open("embedding_meta_"+str(e)+".json", "w") as f:
				json.dump(embedding_meta, f)
			loss_tracker.save_loss("training_loss_"+str(e)+".json")

			embedding = None
			embedding_saver = None
			embedding_meta = None
			
		
	print("Finishing training, saving loss")
	
	
			


def test(net, embedding, embedding_meta):
	"""
	1. compute the embedding of the test image
	2. compute the eculidean distance bewteen test embedding and all the embedding in the training set
	3. ...
	"""
	net.eval()
	accuracy = 0.
	time_init = time.time()
	embedding_saver = []
	embedding_meta_saver = {"label": [], "image_path": []}
	with torch.no_grad():
		for i, data in enumerate(test_loader):
			image, label = data['image'], data['label']
			image = image.to(device)
			output = net(image)
			output = output.cpu()
			# print(image.type)
			# print(embedding.shape)
			embedding_saver.append(output)
			
			embedding_meta_saver["label"] += label
			embedding_meta_saver["image_path"] += data['image_path']
			#print("label size : ")
			#print(embedding_meta["label"].shape)
			#_, idx = embedding.query(output, k=30)
			#print(idx.shape)
			# print(idx)
			#pred = embedding_meta["label"][idx]
			#print(pred.shape)
			
			#for b in range(opt.bs):
				# print(np.sum(pred[b]==label[b])/30)
				#accuracy += np.sum(pred[b]==label[b])/30
				# print(pred[b] == label[b])					
			# print("Accuracy:")
			#print(accuracy/((i+1)*opt.bs))
			progress_bar(i, len(test_loader))
		#print("Test Acuracy: %.2f" % (accuracy/((i+1)*opt.bs)*100))
		torch.save(torch.cat(embedding_saver), "embedding_test.pth")
		with open("embedding_meta_testing.json", "w") as f:
			json.dump(embedding_meta_saver, f)
		print('Total time: ', time.time()-time_init)




def main():
	print(opt)
	lr = opt.lr
	net = models.resnet50(pretrained=True)
	# counter = 1
	# for name, param in net.named_parameters():
	# 	print("%d: %s" %(counter, name) )
	# 	if counter >= 31:
	# 		break
	# 	param.requires_grad = False
	# 	counter += 1

	num_ftrs = net.fc.in_features
	net.fc = nn.Linear(num_ftrs, 4096)
	# net.load_state_dict(torch.load("./checkpoint_39.pt"))
	#loss_tracker.load_loss("./training_loss_12.json")
	net = net.to(device)
	optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
	criterion = nn.TripletMarginLoss()
	
	# summary(net,input_size=(3, 224, 224))
	# print(os.listdir())
	print("training now")
	train(net, criterion, optimizer, 50)
	print("loading embedding")
	#embedding = torch.load("embedding_6.pth").detach().numpy()#.to(device)
	# assert(torch.equal(embedding, torch.cat(embedding_saver)))
	#with open("embedding_meta_6.json", "r") as f:
	#	embedding_meta = json.load(f)
	#print("converting to KDTree")
	
	#embedding_tree = neighbors.KDTree(embedding)
	#embedding = None
	#embedding_meta["label"] = np.array(embedding_meta["label"])
	
	print("starting to test")
	test(net, None, None)


if __name__ == "__main__":
	main()
