# Final project for DL

Due Dates
---------
Presentation: <b>May 15th</b>

Writeup: <b>May 5th</b>


Notes
=====

VGG paper: https://arxiv.org/pdf/1409.1556.pdf
Resnet paper: https://arxiv.org/pdf/1512.03385.pdf


Resnet paper does not use dropout because it is very deep (form of regularization, see paragraph before 4.3). Our implementation is small and may require dropout.
Resnet paper uses batchnorm before relu. I guess it's ok, order doesn't really matter...

stride=2 -> divide by 2, kernel size=7 -> subtract (7-1)/2=3
64x64 -> 29x29

note: in minivgg modify convolutions to have zero padding to keep dimenesionality.

For creating own dataset, and for creating torch tensor from PIL image, use torchvision.transforms stuff.

For viewing rgb images, matplotlib expects MxNx3 input.
Example: image is saved as 3x64x64 tensor. Matplotlib requires plt.imshow(image.permute(1,2,0).numpy()) (and followed by plt.show()) to view the image.
