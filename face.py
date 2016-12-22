''' Rasperry pi implementation of face detection via skin segmentation of the pper
"A simple and accurate face detection algorithm in complex background"
by Yu-Tang Pai, Shanq-Jang Ruan, Mon-Chau Shie, Yi-Chi Liu 

Author: Chibueze Ukachi''' 


#numpy gives matlab matrix functionality
import numpy as np
#PILLOW allows us to open and show the images, aswell as draw rectangles
from PIL import Image
from PIL import ImageDraw

#open image file
#img = Image.open('ron.jpg')
img = Image.open('nadal.jpg')
#convert to numpy array
img = np.asarray(img)
original_image = img
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
#Skin - 1
#Non-skin - 0
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

bw = np.uint8(skin_wnr)

#downsampling is needed as the connected component labelling is recursive 
#python crashes in matrix is too large
down_sample = 5
bw = bw[::down_sample,::down_sample]

#collect image size after downsampling
bw_row,bw_col = bw.shape

'''I manually ported the connected component labelling from Matlab to python with slight modifications
Credit to :michael scheinfeild
https://ch.mathworks.com/matlabcentral/fileexchange/38010-connected-component-labeling-like-bwlabel
'''
def findConnectedLabels(L,startLabel,bwcur,ir,ic,m,n):

    a = bwcur[ir+1, ic]
    b = bwcur[ir-1, ic]   
    c = bwcur[ir, ic+1] 
    d = bwcur[ir, ic-1]

    aa = L[ir+1, ic]
    bb = L[ir-1, ic]
    cc = L[ir, ic+1] 
    dd = L[ir, ic-1]

    if((a==1) and (aa==0)):
        L[ir+1, ic]=startLabel
        L  = findConnectedLabels(L,startLabel,bwcur,ir+1,ic,m,n)

    if((b==1) and (bb==0)):
        L[ir-1, ic]=startLabel
        L  = findConnectedLabels(L,startLabel,bwcur,ir-1,ic,m,n)

    if((c==1)and(cc==0)):
        L[ir, ic+1]=startLabel
        L  = findConnectedLabels(L,startLabel,bwcur,ir,ic+1,m,n)

    if((d==1)and (dd==0)):
        L[ir, ic-1]=startLabel
        L  = findConnectedLabels(L,startLabel,bwcur,ir,ic-1,m,n)


    return L

def detect(bw):
    
    m,n = bw.shape
    emp1= np.zeros((1,n),dtype = int)
    t1 = np.append(emp1,bw,axis=0)
    bwnnew = np.append(t1,emp1,axis=0)

    emp2 = np.zeros((m+2,1),dtype=int)
    t2 = np.append(emp2,bwnnew,axis=1)
    bwnnew = np.append(t2,emp2,axis=1)


    L = np.zeros(bwnnew.shape,dtype = int)

    startLabel = 1

    for ir in range(2,m+2):
        for ic in range(2,n+2):
            curdata = bwnnew[ir,ic]
            #print(curdata)
            lc = L[ir,ic]
            if (curdata ==1) and(lc ==0):
                L[ir,ic]= startLabel
                L = findConnectedLabels(L,startLabel,bwnnew,ir,ic,m,n);
                startLabel = startLabel  + 1
                
    
    return L[1:m+1,1:n+1]
    
#I added face prop as really small groups of peoples that are connected are unlikely to be a face
face_prop = 0.002 *bw_row*bw_col

#collecting the labels
bw_label = detect(bw)


#note that after the connected component labelling there can be a huge number of labels 
#as isolated pixels are given labels 
final_label = []
start_pos_x = []
start_pos_y = []
height_list = []
width_list = []

#these are from tests that I performed. 
low_threshold = 1.0
high_threshold = 1.5

#for every label
for label in range(1,np.amax(bw_label)):
    #ignore it if it could be a smudge
    if len(bw_label[bw_label == label]) > face_prop:
        temp_bw_label = np.array(bw_label, copy=True)
        temp_bw_label[temp_bw_label!=label] = 0
        row, col = np.where(temp_bw_label==label)
        #these are the positions required for the bounding box
        start_x = np.amin(col)
        start_y = np.amin(row)
        width = np.amax(col) -  start_x + 1
        height = np.amax(row) - start_y + 1
        
        #height to width ratio used to choose label
        if ((height/width > low_threshold) and (height/width <high_threshold)):
            #tcurrent label number
            final_label.append(label)
            #these are the positions required for the bounding box for each iteration
            start_pos_x.append(start_x)
            start_pos_y.append(start_y)
            height_list.append(height)
            width_list.append(width)

for i in range(len(final_label)):
    label_num = final_label[i]
    start_x = start_pos_x[i]
    start_y = start_pos_y[i]
    height = height_list[i]
    width = width_list[i]
    img1 = Image.fromarray(original_image)
    draw = ImageDraw.Draw(img1)
    
    draw.rectangle([down_sample*start_x,down_sample*start_y,down_sample*(start_x+width),
                    down_sample*(start_y+height)], fill=None, outline="red")
    img1.show()
    
