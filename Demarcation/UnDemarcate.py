import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
def build_struct(h,w):
    if(h<=1 or w<=1):
        h=3
        w=1
    return np.ones((h,w))

def zero_pad(img,struct_elm):
    img_height=img.shape[0]
    img_width=img.shape[1]
    kernel_height=struct_elm.shape[0]
    kernel_width=struct_elm.shape[1]
    row=kernel_height//2
    col=kernel_width//2
    padded=np.zeros((img_height+2*(kernel_height//2),img_width+2*(kernel_width//2)),dtype=np.uint8)
    padded[row:row+img_height,col:col+img_width]=img
    return padded

def display_image(title,img):
    cv2.imshow(title,img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def fit(struct_elm,img_part): #for erosion
    sum=0
    output=0
    size=struct_elm.shape[0]*struct_elm.shape[1]
    for i in range (struct_elm.shape[0]):
        for j in range(struct_elm.shape[1]):
            sum=sum+(struct_elm[i,j])*(img_part[i,j])
    sum=sum/255
    if(sum==size):
        output=1
    return output

def hit(struct_elm,img_part): 
    sum=0
    output=0
    for i in range (struct_elm.shape[0]):
        for j in range(struct_elm.shape[1]):
            sum=sum+(struct_elm[i,j])*(img_part[i,j])
    if(sum>0):
        output= 1
    return output

def convolute(struct_elm,img,opp):
    output=np.zeros(img.shape)
    padded=zero_pad(img,struct_elm)
    for j in range (img.shape[1]):
        for i in range(img.shape[0]):
            img_part=padded[i:i+struct_elm.shape[0],j:j+struct_elm.shape[1]]
            output[i,j]=opp(struct_elm,img_part)
    return output 

def erosion(struct_elm,img):
    img=convolute(struct_elm,img,fit)
    return img

def dilation(struct_elm,img):
    img=convolute(struct_elm,img,hit)
    return img

def reverse(img):
    return 1-img


#MAIN PART
src=sys.argv[1]
if(len(sys.argv)>2):
    output_flag=sys.argv[2]
    output_file=sys.argv[3]
img=cv2.imread(src)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
display_image('Original',img)
_,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
max_size=0
dev=0
max_height=0
max_width=0
for contour in contours:
    _,_,width,height=cv2.boundingRect(contour)
    if(max_height<height):
        max_height=height
    if(max_width<width):
        max_width=width
min_size=max_height*max_width
se_h=1
se_w=1
for contour in contours:
    _,_,width,height=cv2.boundingRect(contour)
    if(width/max_width<0.9 and height/max_height<0.5):
        area=cv2.contourArea(contour)
        if(min_size>area):
            min_size=area
            se_h=height
            se_w=width
struct_elm=build_struct(se_h,se_w)
img=erosion(struct_elm,img)
img=reverse(img)
img=img*255
_,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
cleanup_se=np.ones((1,1))
img=erosion(cleanup_se,img)
img=dilation(struct_elm,img)
img=reverse(img)
display_image('UnDemarcated',img)
if(len(sys.argv)>2):
    cv2.imwrite(output_file,img*255)

