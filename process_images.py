
"""
tiny imagenet:
200 classes
500 training image for each class
50 validation images for each class
50 test images for each class
"""

from PIL import Image
import torch
import torchvision


top_dir = "../tiny-imagenet-200/"
wnids = top_dir + "wnids.txt"
train_dir = top_dir + "train/"
val_dir = top_dir + "val/"
test_dir = top_dir + "test/"
train_images_path = top_dir + "train_images" # contains tensor-ized images
val_images_path = top_dir + "val_images" # contains tensor-ized images
test_images_path = top_dir + "test_images" # contains tensor-ized images

image_class_dict = {} # directory name : label (i.e. n03444034 : 35)
val_counter = {} # the data set structure and naming scheme sucks

# get name of image directories, fill dict
all_dirs = []
with open(wnids) as file:
	i = -1
	for line in file:
		i += 1
		dir_name = line[:-1]
		all_dirs.append(dir_name)
		image_class_dict[dir_name] = i
		val_counter[dir_name] = 0


pil_to_tensor = torchvision.transforms.ToTensor() # convert PIL image to float tensor
make_gray = torchvision.transforms.Grayscale() # convert PIL image to grayscale PIL image

# convert train images to tensor
train_images = torch.FloatTensor(200, 500, 64, 64) # 200 classes, 500 examples, image dims
for curr_dir in all_dirs:
	image_class  = image_class_dict[curr_dir]
	for i in range(500):
		image_file = train_dir + curr_dir + "/images/" + curr_dir + "_" + str(i) + ".JPEG"
		im = Image.open(image_file)
		train_images[image_class, i] = pil_to_tensor(make_gray(im)).squeeze(0)

torch.save(train_images, train_images_path) # save the tensor, note that dim=0 is the label


# convert val images to tensor
val_images = torch.FloatTensor(200, 50, 64, 64) # 200 classes, 50 examples, image dims
with open(val_dir + "val_annotations.txt") as ann_file:
	i = -1
	for line in ann_file:
		i += 1
		image_file = val_dir + "images/val_" + str(i) + ".JPEG"
		im = Image.open(image_file)
		dir_name = line.split()[1]
		label = image_class_dict[dir_name]
		curr_example = val_counter[dir_name]
		val_images[label, curr_example] = pil_to_tensor(make_gray(im)).squeeze(0)
		val_counter[dir_name] += 1 # counter should be 50 for all classes when finished

torch.save(val_images, val_images_path) # save the tensor, note that dim=0 is the label

# convert test images to tensor
test_images = torch.FloatTensor(10000, 64, 64) # 10000 images, image dims
for i in range(10000):
	image_file = test_dir + "images/test_" + str(i) + ".JPEG"
	im = Image.open(image_file)
	test_images[i] = pil_to_tensor(make_gray(im)).squeeze(0)

torch.save(test_images, test_images_path) # save the tensor, note that dim=0 is the label
