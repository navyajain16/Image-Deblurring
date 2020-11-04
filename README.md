# Deblurring images in python

Recently,as I was going through my gallery I found many images that were pretty good but were blurred. This got me to think if there was some way to remove blur from these images. 

Since I am learning python these days so I decided to use my knowledge of python to remove blur from images.

As I was looking for ways, I came across a word kernel which played an important role in deblurring. I wondered what exactly is a kernel?

## What is Kernel?

A kernel, convolution matrix,or mask in image processing is a small matrix that is used for various applications like blurring, sharpening, embossing, edge detection, etc which is done by a convolution between an image and a kernel.

## 2D CONVOLUTION IN IMAGES 

While reading about kernel I saw the word Convolution and a question popped into my mind that what is convolution and how is it applied to images?

Convolution, a simple mathematical operation, is a way of `multiplying together' two arrays of numbers having different sizes but the same dimensionality to produce a third array of numbers of the same dimensionality. 

When convolution operation is applied to an image, then it is can be said that a simple mathematical operation is done over the image. The values of the pixels in the image are changed to some degree during convolution operation. 
kernel or the filter is used to carry out convolution operation .

<img src="https://github.com/navyajain16/navyajain16/blob/main/image/figure1.jpg" width=200 /> 

For example, it can be seen in the figure that a 3×3 kernel is applied over a 7×7 dimensional image. By taking the values of the kernel into consideration, we can change the values of the image pixels.

By using the identity kernel, as given below, for carrying out the convolution process on an image, as a result, we get an image that is the same as the original image. 

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure2.jpg)

Note that by using a kernel, detecting and highlighting of edges, sharpening, and un sharpening images can be done.

## DEBLURRING OF IMAGES

After learning some basic concepts about deblurring, I looked into ways to deblur an image in python. I found many methods to deblur like by Lucy Richardson Algorithm, using Wiener filter, Sharpening filter, etc. Among all these methods the one I liked the most was deblurring by sharpening filter which is there in the open CV.

I personally was able to understand it the best so I decided to proceed with it.

## DEBLURRING OF IMAGES BY SHARPENING FILTER 

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/roseblur.jpg) &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; ![Image](https://github.com/navyajain16/navyajain16/blob/main/image/rosesharpen.jpg) &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;           ![Image](https://github.com/navyajain16/navyajain16/blob/main/image/rosedenoise.jpg)

   *Blurred Image* &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    *Sharpened Image*&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    *Denoised Image*
   
For deblurring images by this method there are 2 steps to be followed:
1.	The first is to sharpen the image. 
2.	Then denoise it to remove noise from the image.

Let’s start with deblurring.
I have deblurred images using open CV library.

## PREREQUISITES FOR DEBLURRING USING OPEN CV 

### LIBRARIES 

Before starting with deblurring using open cv make sure you have installed the following libraries 

1.	Open CV: It is a Python library that can be used to solve computer vision problems. 
         This library can be installed by writing the following code in command prompt or anaconda prompt: Pip install opencv-python
         It is imported as import cv2 in the code.
         
2.	NumPy: NumPy is a python library that is used for working with arrays. It also has functions for working in the domain of linear algebra, Fourier transform, and matrices.
This library is usually pre-installed in python (Anaconda 3). But if it’s not there then it can be installed by writing the following code in command prompt or anaconda prompt: Pip install numpy.

It is imported as import numpy in the code.

## DEBLURRING OF IMAGES 

The code I used to deblur my images is:  

### CODE

```markdown
import cv2
import numpy as np

image = cv2.imread('flower.jpg')

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(image, 0 , sharpen_kernel)

deblurred = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)

cv2.imshow(‘deblureed’, deblurred)
cv2.waitKey ()
```





### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/navyajain16/navyajain16.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
