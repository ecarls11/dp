
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
import sys

gray = False
ROTATE=False




top_dir = "../tiny-imagenet-200/"
wnids = top_dir + "wnids.txt"
train_dir = top_dir + "train/"
val_dir = top_dir + "val/"
test_dir = top_dir + "test/"
train_images_path = top_dir + "train_images" # contains tensor-ized images
train_labels_path = top_dir + "train_labels" # contains tensor-ized labels
val_images_path = top_dir + "val_images" # contains tensor-ized images
val_labels_path = top_dir + "val_labels" # contains tensor-ized images

image_class_dict = {} # directory name : label (i.e. n03444034 : 35)
val_counter = {}

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

if len(sys.argv) > 1:
    ROTATE = True
    train_images_path = top_dir + "train_images" + ".rotated" # contains tensor-ized images
    train_labels_path = top_dir + "train_labels" + ".rotated" # contains tensor-ized labels
    val_images_path = top_dir + "val_images" + ".rotated" # contains tensor-ized images
    val_labels_path = top_dir + "val_labels" + ".rotated" # contains tensor-ized images


pil_to_tensor = torchvision.transforms.ToTensor() # convert PIL image to float tensor
make_gray = torchvision.transforms.Grayscale() # convert PIL image to grayscale PIL image
y = torchvision.transforms.RandomRotation(360, center=(32,32))
hflip = torchvision.transforms.RandomHorizontalFlip(0.5)
vlfip = torchvision.transforms.RandomVerticalFlip(0.5)


if gray:
    # convert train images to tensor
    offset = 0
    train_images = torch.zeros(100000, 1, 64, 64) # 200 classes, 500 examples, image dims
    train_labels = torch.zeros(100000).long()
    for curr_dir in all_dirs:
        image_class  = image_class_dict[curr_dir]
        for i in range(500):
            image_file = train_dir + curr_dir + "/images/" + curr_dir + "_" + str(i) + ".JPEG"
            im = Image.open(image_file)
            if ROTATE: im = y(im)
            train_images[i + offset] = pil_to_tensor(make_gray(im))
            train_labels[i + offset] = image_class
        offset += 500
    train_images = (train_images - train_images.mean()) / train_images.std()
    torch.save(train_images, train_images_path)
    torch.save(train_labels, train_labels_path)


    # convert val images to tensor
    val_images = torch.zeros(10000, 1, 64, 64) # 200 classes, 50 examples, image dims
    val_labels = torch.zeros(10000).long()
    with open(val_dir + "val_annotations.txt") as ann_file:
        i = -1
        for line in ann_file:
            i += 1
            image_file = val_dir + "images/val_" + str(i) + ".JPEG"
            im = Image.open(image_file)
            if (ROTATE):
                im = y(im)
            dir_name = line.split()[1]
            label = image_class_dict[dir_name]
            curr_example = val_counter[dir_name]
            val_images[i] = pil_to_tensor(make_gray(im))
            val_labels[i] = label
            val_counter[dir_name] += 1 # counter should be 50 for all classes when finished
    val_images = (val_images - val_images.mean()) / val_images.std()
    torch.save(val_images, val_images_path)
    torch.save(val_labels, val_labels_path)



else:
    # convert train images to tensor
    offset = 0
    train_images = torch.zeros(100000, 3, 64, 64) # 200 classes, 500 examples, image dims
    train_labels = torch.zeros(100000).long()
    for curr_dir in all_dirs:
        image_class  = image_class_dict[curr_dir]
        for i in range(500):
            image_file = train_dir + curr_dir + "/images/" + curr_dir + "_" + str(i) + ".JPEG"
            im = Image.open(image_file)
            if random.random() >= 0.5:
                im = y(im)
            if random.random() >= 0.5:
                k = torchvision.transforms.ColorJitter(random.random(), random.random(), random.random(), random.random()*0.5)
                im = k(im)
            im = hflip(im)
            im = vflip(im)
            im = pil_to_tensor(im)
            if im.size()[0] == 3:
                train_images[i + offset] = im
                train_labels[i + offset] = image_class
        offset += 500
    train_images = (train_images - train_images.mean()) / train_images.std()
    torch.save(train_images, train_images_path)
    torch.save(train_labels, train_labels_path)

    # convert val images to tensor
    val_images = torch.zeros(10000, 3, 64, 64) # 200 classes, 50 examples, image dims
    val_labels = torch.zeros(10000).long()
    with open(val_dir + "val_annotations.txt") as ann_file:
        i = -1
        for line in ann_file:
            i += 1
            image_file = val_dir + "images/val_" + str(i) + ".JPEG"
            im = Image.open(image_file)
            im = pil_to_tensor(im)
            dir_name = line.split()[1]
            label = image_class_dict[dir_name]
            curr_example = val_counter[dir_name]
            if im.size()[0] == 3:
                val_images[i] = im
                val_labels[i] = label
                val_counter[dir_name] += 1 # counter should be 50 for all classes when finished
    val_images = (val_images - val_images.mean()) / val_images.std()
    torch.save(val_images, val_images_path)
    torch.save(val_labels, val_labels_path)
