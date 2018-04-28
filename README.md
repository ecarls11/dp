TODO

Read in image net data as grayscale.
Implement tiny vggnet.
Implement tiny resnet.

Resnet paper does not use dropout because it is very deep (form of regularization, see paragraph before 4.3). Our implementation is small and may require dropout.
Resnet paper uses batchnorm before relu. I guess it's ok, order doesn't really matter...

stride=2 -> divide by 2, kernel size=7 -> subtract (7-1)/2=3
64x64 -> 29x29

note: in minivgg modify convolutions to have zero padding to keep dimenesionality.

For creating own dataset, use torchvision.transforms stuff.