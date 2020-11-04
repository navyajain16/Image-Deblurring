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

<img src="https://github.com/navyajain16/navyajain16/blob/main/image/figure1.jpg?raw=true" width=400 /> 

For example, it can be seen in the figure that a 3×3 kernel is applied over a 7×7 dimensional image. By taking the values of the kernel into consideration, we can change the values of the image pixels.

By using the identity kernel, as given below, for carrying out the convolution process on an image, as a result, we get an image that is the same as the original image. 

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure2.jpg)

Note that by using a kernel, detecting and highlighting of edges, sharpening, and un sharpening images can be done.

## DEBLURRING OF IMAGES

After learning some basic concepts about deblurring, I looked into ways to deblur an image in python. I found many methods to deblur like by Lucy Richardson Algorithm, using Wiener filter, Sharpening filter, etc. Among all these methods the one I liked the most was deblurring by sharpening filter which is there in the open CV.

I personally was able to understand it the best so I decided to proceed with it.

## DEBLURRING OF IMAGES BY SHARPENING FILTER 

<img src="https://github.com/navyajain16/navyajain16/blob/main/image/roseblur.jpg" width=250 />  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://github.com/navyajain16/navyajain16/blob/main/image/rosesharpen.jpg" width=250 />  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://github.com/navyajain16/navyajain16/blob/main/image/rosedenoise.jpg" width=250 /> 

   *Blurred Image* &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   *Sharpened Image*&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   *Denoised Image*
   
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
## LINE BY LINE EXPLANATION OF CODE 

```markdown
import cv2
import numpy as np
```
I have imported library cv2 and numpy in the above line as mentioned above.

```markdown
image = cv2.imread('flower.jpg')
```
I have used cv2.imread() method in the code.     

cv2.imread() is a method in python that is used for loading an image from the specified file\\.

Here the image loaded is “flower.jpg” which is the image to be blurred. “img” is the variable here. Its syntax is as follows:

_cv2.imread(path, flag)_

**Parameters:**

**path:**  Path is a string that represents the path of the image to be read.

**flag:** Flag specifies how the image is read. The default value is cv2.IMREAD_COLOR

```markdown
sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
```

This is the convolution kernel which will be used for sharpening the image. The explanation of kernel is mentioned above. 

```markdown
sharpen = cv2.filter2D(image , 0 , sharpen_kernel)
```

By use of this line of code, I have sharpened the image.                           
In Open CV Sharpen filter doesn’t have an implemented function. Therefore, cv2.filter2D() function is used which processes the with an arbitrary filter. 
Colored images are often sharpened with this function. This operation convolves an image with the kernel. Its syntax is as follows:

_filter2D(src, dst, ddepth, kernel)_

**Parameters:**
**src:** It is the link of image to be deblurred.
**dst** − It is the output image of the same size and type as src.   
**ddepth** – An integer type variable representing the depth of the output image.
**kernel** – It is the convolution kernel.

This is what a sharpened image would look like:

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/rosesharpen.jpg)

*Sharpened image*

```markdown
deblurred = cv2.fastNlMeansDenoisingColored(gausBlur,None,10,10,7,21)
```

By this line of code, I have denoised the image to remove noise from a sharpened image.                                                                          cv2.fastNlMeansDenoisingColored() function is the of Non-local Means Denoising algorithm implementation. 
It is mostly used to remove Gaussian noise. Its syntax is as follows

_cv2.fastNlMeansDenoisingColored( src [, dst [ , h [, hcolor [, templateWindowSize [, searchWindowSize ]]]]])_

**Parameters:** 
**src** : It is the link of image to be deblurred.                                                      
**dst** : Output image with the same size and type as src.                              
**h**: filter strength is regulated by this component. The greater h component greater the image denoised but also removes noise from minute details and smaller the h component smaller the image denoised and also preserves the noise in minute details.                                                                   
**templateWindowSize**: Template patch size in pixels that are used to compute weights. It Should be odd preferably 7 pixels. 
**searchWindowSize** : Size of the window in pixels that can compute the weighted average for a given pixel. Just like templateWindowSize this should be odd too preferably 21 pixels. It affects the performance linearly that is greater searchWindowsSize, greater will be denoising time. 

This is what denoised image would look like :   

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/rosedenoise.jpg)

*Denoised image*

```markdown
cv2.imshow('sharpen', deblurred)
```

We have used cv2.imshow() function above 
To display an image in a window, cv2.imshow() method is used.    
Using this method we have shown the deblurred image i.e. the output image. Its syntax is as follows:

_cv2.imshow(window_name, image)_

**Parameters:**
**window_name**: This represents the name of the window in which the image will be displayed. 
**image**: It is the output displayed image.

```markdown
cv2.waitKey(0)
```

We have used cv2.waitKey() function.                                                    
It is a function used for keyboard binding. Time in milliseconds is its arguments. For specified milliseconds this for any keyboard event. The program is continued if any key is pressed at that time. It waits indefinitely for a keystroke in case 0 is passed.


## OUTPUT IMAGE
Following is the output of the code above:

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/rosedenoise.jpg)

*Deblurred image*

After deblurring the image, I wondered if the image I deblurred is similar to the original one or if it is different from the original image. If this algorithm was right or not. So I decided to take an original picture blur it and then again deblur it to confirm my method.

I choose the most popular image processing image Lena for this purpose

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lena2.jpg)

Then I started to find ways to blur an image. 

## TYPES OF BLURRING 

I found that there were 4 kinds of blurring methods:
•	Average Blur also referred to as box filter or average filter
•	Gaussian Blur also referred to as Gaussian filter
•	Median Blur also referred to as Median filter
•	Bilateral Blur also referred to as Bilateral filter

Average Blur and Gaussian blur are the most commonly used blurring techniques about which I tried and have discussed below:

## PREREQUISITES FOR BLURRING USING OPEN CV

## LIBRARIES

Before starting with blurring using open cv make sure you have installed the following libraries

1.	Open CV: It is a Python library that can be used to solve computer vision problems. 
         This library can be installed by writing the following code in command prompt or anaconda prompt: Pip install opencv-python
         It is imported as import cv2 in the code

## 1.	Average Blur

In the picture below we can see that the input image on the left is processed with the averaging filter (box filter). 

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lena2.jpg) &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lenaavg.jpg)

*Original image*  &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Average blur*

It is the basic blur filter. Here, it has the same value of 1/9 for all coefficient values. On applying the convolution operator, we get an output the same as on the right side as shown. The image will be more blurred as a filter size increases.  

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure3.jpg)

This is the code of average filter which I tried :

### CODE 

```markdown
import cv2

img = cv2.imread("lena.jpg")

blurImg = cv2.blur(img,(9,9))

cv2.imshow('Averaging',blurImg )

cv2.destroyAllWindows()
```

## LINE BY LINE EXPLANATION OF CODE 

```markdown
import cv2
```

I have imported library cv2 in the above line as mentioned above.

```markdown
img = cv2.imread("lena.jpg")
```

I have used cv2.imread() method in the code.                                             
The explanation of this method is given above.

```markdown
blurImg = cv2.blur(image,(9,9))
```
                      
I have used cv2.blur() method in the code. 
Using this method an image is blurred using the normalized box filter. The kernel which is represented below is used by the function to smooth an image:

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure4.jpg)

                                                                         
This the function that blurs the image by the average filter. Its syntax is as follows:

_cv2.blur(src, ksize[, dst[, anchor[, borderType]]])_

**Parameters:**

**src**: it's the link of the image which is to be blurred.

**ksize**: A blurring kernel size tupple.

**dst**: It is the output image of the identical size and type as src.

**anchor**: An integer type variable used to represent anchor point. The point is (-1, -1) is its default value which implies that the anchor is at the kernel center.

**borderType**: It is for type border to be added. It is defined by flags like cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, etc

```markdown
cv2.imshow('Averaging',blurImg )
```

I have used cv2.imshow() function above 
The explanation of this function is given above.

```markdown
cv2.waitKey(0)
```

I have used cv2.waitKey() function.                                                            
The explanation of this function is given above.

```markdown
cv2.destroyAllWindows()
```

I have used cv2.destroyAllWindows() function.  Using this function all the windows we created are destroyed.

## OUTPUT IMAGE

This is the output I got after running the code:

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lenaavg.jpg)

*Average blur*

## 2.	Gaussian Blur 
The Gaussian filter or blur is a slightly more complicated filter and is the most commonly used kernel in image processing It is called the Gaussian filter.
In the picture below we can see that the input image on the left is processed with the Gaussian Filter shown in right. 
                     
![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lena2.jpg) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lenagaus.jpg)

*Original Image* &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *Gaussian Blur* 

The famous Gaussian function is used for creation of this filter. 
The probability that events are centered around a mean value is represented by this function. Furthermore, the width of this function is controlled by the standard deviation (σ) of this function. Sampling this function’s values, gives the coefficients for a Gaussian filter matrix. Effect of different (σ) values can be observed in the following image.

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure5.jpg)


<img src="https://github.com/navyajain16/navyajain16/blob/main/image/figure6.jpg" width=400/>


The equation for a Gaussian filter kernel of size (2k+1)×(2k+1) is given by:

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure7.jpg)

A 5x5 gaussian filter will look like this :

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/figure8.jpg)


### CODE 

```markdown
import cv2

img = cv2.imread("lena.jpg")

gausBlur = cv2.GaussianBlur(image, (5,5),0)

cv2.imshow('Gaussian Blurring', gausBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## LINE BY LINE EXPLANATION OF CODE

All lines of code are explained except line 3 which is explained below.

```markdown
gausBlur = cv2.GaussianBlur(image, (5,5),0)
```

I have used cv2.GaussianBlur() function for Gaussian blurring.      
In this function, the width and height of the kernel specified should be positive and odd. The standard deviation in the X and Y directions should also be specified ascsigma X and sigma Y respectively. If only sigma X is specified, sigma Y is taken as equal to sigma X. Its syntax is as follows:

_cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )_

**Parameters:**
**src**: it's the link of image which is to be blurred.
**ksize**: It is the Gaussian Kernel Size. If ksize is set to [0 0], then ksize is computed from sigma values. 
**dst**: It is the output image of the same size and type as src.
**sigma X**: Kernel standard deviation along X-axis (horizontal direction).                                                                                             
**sigma Y**: Kernel standard deviation along Y-axis (vertical direction). If sigma Y=0, then sigma X value is taken for sigma Y
**borderType**: It depicts what kind of border to be added. It is defined by flags like cv2.BORDER_CONSTANT, cv2.BORDER_REFLECT, etc.

## OUTPUT IMAGE
This is the output I got after running the above code :

![Image](https://github.com/navyajain16/navyajain16/blob/main/image/lenagaus.jpg)

*Gaussian Blur*

## DEBLURRING OF IMAGES

After learning about blurring I tried deblurring them again to verify the results of the algorithm used. This is what I did :

### CODE 

```markdown
import cv2
import numpy as np

image = cv2.imread('lena2.jpg')

gausBlur = cv2.GaussianBlur(image, (5,5),0)

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
sharpen = cv2.filter2D(gausBlur, 0 , sharpen_kernel)

deblurred = cv2.fastNlMeansDenoisingColored(sharpen,None,10,10,7,21)

cv2.imshow('deblurred', deblurred)
cv2.waitKey()
```

## RESULTS

After running this code these are the results I got. As it can be seen the original image and deblurred final images are very similar. Hence the method I used was successful for deblurring of images.

<img src="https://github.com/navyajain16/navyajain16/blob/main/image/lena2.jpg" width=150/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://github.com/navyajain16/navyajain16/blob/main/image/lenagaus.jpg" width=150/> &nbsp;&nbsp;&nbsp;&nbsp;  <img src="https://github.com/navyajain16/navyajain16/blob/main/image/lenasharp.jpg" width=150/> &nbsp;&nbsp;&nbsp;&nbsp; <img src="https://github.com/navyajain16/navyajain16/blob/main/image/lenadenoise.jpg" width=150/>

## REFERENCES

For Learning about image deblurring and blurring I referred to the following websites and articles. These articles helped me a lot in learning deblurring and blurring of images. 

[https://www.analyticsvidhya.com/blog/2020/09/how-to-perform-blur-detection-using-opencv-in-python/](https://www.analyticsvidhya.com/blog/2020/09/how-to-perform-blur-detection-using-opencv-in-python/)

[https://homepages.inf.ed.ac.uk/rbf/HIPR2/convolve.htm](https://homepages.inf.ed.ac.uk/rbf/HIPR2/convolve.htm)

[https://debuggercafe.com/image-and-video-blurring-using-opencv-and-python/](https://debuggercafe.com/image-and-video-blurring-using-opencv-and-python/)

[https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html)

[http://datahacker.rs/004-how-to-smooth-and-sharpen-an-image-in-opencv/](http://datahacker.rs/004-how-to-smooth-and-sharpen-an-image-in-opencv/) 

## CONCLUSION 

This is what I did to deblur images. It was a successful method but as said above there are many more methods to deblur images that can be tried out.
I hope this would help you also to deblur some images of your own.







         














