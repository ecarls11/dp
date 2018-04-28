from PIL import Image
import torch
import torchvision

top_dir = "../tiny-imagenet-200/"
wnids = top_dir + "wnidts.txt"

train_dirs = []
with open(wnids) as file:
	for line in file:
		train_dirs += line

print (len(train_dirs))