''' Rasperry pi implementation of the paper:
"A simple and accurate face detection algorithm in complex background"
by Yu-Tang Pai, Shanq-Jang Ruan, Mon-Chau Shie, Yi-Chi Liu ''' 
#numpy gives matlab matrix functionality
import numpy as np
#PILLOW allows us to open and show the images
from PIL import Image

#img = Image.open('ron.jpg')
img = Image.open('nadal.jpg')
img = np.asarray(img)
img.astype(dtype = 'double')

#retrieving RGB components
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]

#height, width and depth
[height,width,depth] = img.shape

#function to convert from rgb to YCbCr
def YCbCr(r, g, b): 
    y = 0.3*r + 0.6*g + 0.1*b
    cb = 0.2*r -0.3*g + 0.5*b
    cr = 0.5*r - 0.4*g - 0.1*b
    return y, cb, cr

y,cb,cr = YCbCr(red,green,blue)    

#ensure that y is normalised between 0 and 255
def normaliser(y):
    norm_y = 255*((y-np.amin(y))/(np.amax(y)-np.amin(y)))
    return norm_y


norm_y = normaliser(y)    

#compute the average value of y
avg_y = np.sum(y)/(height*width)

#if the luminance is outside a designated range, it will readjusted by the
#following t vlaues
def t_setter(avg_y):
    if (avg_y < 64):
        t = 1.4
    elif (avg_y > 192):
        t = 0.6
    else:
        t = 1
    return t

t = t_setter(avg_y)

#uses the value of t to compensate the different rgb components
if  (t == 1):
    comp_r = red
    comp_g= green
else:
    comp_r = red**t
    comp_g = green**t 

#this performs the compensation for different lighting conditions
light_comp = np.zeros((height,width,depth))
light_comp[:,:,0] = comp_r
light_comp[:,:,1] = comp_g
light_comp[:,:,2] = blue


#skin colour detection by designated chrominance range between 10 and 45
cr[np.where(cr<10) ] =0
cr[np.where(cr>45) ] =0
cr[np.nonzero(cr)]=1

skin = cr

filter_size = 5
#skin with noise removal 
skin_wnr=np.zeros((height,width))
for i in range(1,height+1 - filter_size):
    for j in range(1,width+1 - filter_size):
        temp_sum=np.sum(skin[i:i+4, j:j+4])
        skin_wnr[i:i+filter_size, j:j+filter_size]=(temp_sum>=13);


img = Image.fromarray(np.uint8(img))

img.show(title= 'Original Image')

light_compensation = Image.fromarray(np.uint8(light_comp))

light_compensation.show(title = 'After lighting compensation')

skin = Image.fromarray(np.uint8(255*skin))

skin.show(title = 'Skin detection complete')


skin_with_noise_removal = Image.fromarray(np.uint8(255*skin_wnr))

skin_with_noise_removal.show('After low pass filter')
