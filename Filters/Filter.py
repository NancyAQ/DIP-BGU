import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math

def display_image(title,img):
    cv2.imshow(title,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def pad(kernel,img):
    img_height=img.shape[0]
    img_width=img.shape[1]
    kernel_height=kernel.shape[0]
    kernel_width=kernel.shape[1]
    row=kernel_height//2
    col=kernel_width//2
    if(len(img.shape)>2):
        channels=img.shape[2]
        padded=np.zeros((img_height+2*(kernel_height//2),img_width+2*(kernel_width//2),channels),dtype=np.uint8)
    else:
        padded=np.zeros((img_height+2*(kernel_height//2),img_width+2*(kernel_width//2)),dtype=np.uint8)
    padded[row:row+img_height,col:col+img_width]=img
    #1 left col
    padded[row:img_height+row,0:col]=np.fliplr(img[:,0:col])
    #2 top row
    padded[0:row,col:col+img_width]=np.flipud(img[0:row,:])
    #3 bottom row
    padded[img_height+row:,col:col+img_width]=np.flipud(img[img_height-row:,:])
    #4 right col
    padded[row:img_height+row,img_width+col:]=np.fliplr(img[:,img_width-col:])
    #corners
    padded[0:row,0:col]=np.flipud(np.fliplr(img[0:row,0:col]))
    padded[0:row,img_width+col:]=np.flipud(np.fliplr(img[0:row,img_width-col:]))
    padded[img_height+row:,0:col]=np.flipud(np.fliplr(img[img_height-row:,0:col]))
    padded[img_height+row:,img_width+col:]=np.flipud(np.fliplr(img[img_height-row:,img_width-col:])) 
    return padded

def multiply_sum(kernel,img_part):
    sum=0
    for i in range (kernel.shape[0]):
        for j in range(kernel.shape[1]):
            sum=sum+(kernel[i,j])*(img_part[i,j])
    return sum
def create_gaussian(size,st_dev):
    gaussian=np.zeros((size,size),np.float32)
    for i in range(-(size//2),(size//2)+1):
        for j in range (-(size//2),(size//2)+1):
            norm_factor=1/(2*np.pi*(st_dev**2))
            exponent=np.exp(-(j**2+i**2)/(2*st_dev**2))
            gaussian[i+size//2,j+size//2]=norm_factor*exponent
    return gaussian/np.sum(gaussian)   
    
def convolution(kernel,img):
    if(len(img.shape)>2):
       output=np.zeros_like(img)
    else:
        output=np.zeros(img.shape)
    padded=pad(kernel,img)
    for j in range (img.shape[1]):
        for i in range(img.shape[0]):
            img_part=padded[i:i+kernel.shape[0],j:j+kernel.shape[1]]
            output[i,j]=multiply_sum(kernel,img_part)
    return output 

def min_select(kernel,img):
    flat=img.flatten()
    flat.sort()
    return flat[0]

def median_select(kernel,img):
    flat=img.flatten()
    flat.sort()
    median=len(flat)//2  
    return flat[median]       

def max_select(kernel,img):
    flat=img.flatten()
    flat.sort()
    return flat[len(flat)-1]
    
def GaussianFilter(size,st_dev,img):
    gaussian=create_gaussian(size,st_dev)
    print('The applied filter:')
    print(gaussian)
    return convolution(gaussian,img)

def magnitude(gx,gy): 
    return np.sqrt(gx**2+gy**2)

def EdgeDetect(size,direction,img):
    xkernel=np.zeros((size,size))
    ykernel=np.zeros((size,size))
    horizontal_kernel_three=np.array([[-1.0,0.0,1.0],[-2.0,0.0,2.0],[-1.0,0.0,1.0]])#the gx
    vertical_kernel_three=np.array([[-1.0,-2.0,-1.0],[0.0,0.0,0.0],[1.0,2.0,1.0]])#the gy
    horizontal_kernel_five=np.array([[-1.0,-2.0,0.0,2.0,1.0],[-2.0,-4.0,0.0,4.0,2.0]
                                     ,[-4.0,-8.0,0.0,8.0,4.0],[-2.0,-4.0,0.0,4.0,2.0]
                                     ,[-1.0,-2.0,0.0,2.0,1.0]])
    vertical_kernel_five=np.array([[-1.0,-2.0,-4.0,-2.0,-1.0],[-2.0,-4.0,-8.0,-4.0,-2.0]
                                   ,[0.0,0.0,0.0,0.0,0.0],[2.0,4.0,8.0,4.0,2.0]
                                   ,[1.0,2.0,4.0,2.0,1.0]])
    horizontal_kernel_five=horizontal_kernel_five*(1/8)
    vertical_kernel_five=vertical_kernel_five*(1/8)
    horizontal_kernel_seven=np.array([[-1.0,-4.0,-6.0,0.0,6.0,4.0,1.0],[-2.0,-8.0,-12.0,0.0,12.0,8.0,2.0],
                                      [-3.0,-12.0,-18.0,0.0,18.0,12.0,3.0],[-4.0,-16.0,-24.0,0.0,24.0,16.0,4.0],
                                      [-3.0,-12.0,-18.0,0.0,18.0,12.0,3.0]
                                      ,[-2.0,-8.0,-12.0,0.0,12.0,8.0,2.0]
                                      ,[-1.0,-4.0,-6.0,0.0,6.0,4.0,1.1]])
    vertical_kernel_seven=np.array([[-1.0,-2.0,-3.0,-4.0,-3.0,-2.0,-1.0],
                                    [-4.0,-8.0,-12.0,-16.0,-12.0,-8.0,-4.0],
                                    [-6.0,-12.0,-18.0,-24.0,-18.0,-12.0,-6.0],
                                    [0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                                    [6.0,12.0,18.0,24.0,18.0,12.0,6.0],
                                    [4.0,8.0,12.0,16.0,12.0,8.0,4.0],
                                    [1.0,2.0,3.0,4.0,3.0,2.0,1.0]])
    horizontal_kernel_seven=horizontal_kernel_seven*(1/32)
    vertical_kernel_seven=vertical_kernel_seven*(1/32)
    if(size==3):
        xkernel=horizontal_kernel_three
        ykernel=vertical_kernel_three
    if(size==5):
        xkernel=horizontal_kernel_five
        ykernel=vertical_kernel_five
    if(size==7):
        xkernel=horizontal_kernel_seven
        ykernel=vertical_kernel_seven
    if(len(img.shape)>2):
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    kernel=np.zeros((size,size),np.float32)
    match(direction):
        case 'H':
            gx=convolution(xkernel,img)
            _,output=cv2.threshold(gx,100,255,cv2.THRESH_BINARY)
            print('The applied filter:')
            print(xkernel)
        case 'V':
            gy=convolution(ykernel,img)
            _,output=cv2.threshold(gy,100,255,cv2.THRESH_BINARY)
            print('The applied filter:')
            print(ykernel)
        case 'HV':
            gx=convolution(xkernel,img)
            gy=convolution(ykernel,img)
            combined_edges=magnitude(gx,gy)
            combined_edges=combined_edges.astype(np.uint8)
            _,output=cv2.threshold(combined_edges,100,255,cv2.THRESH_BINARY)
            print('The applied filters:')
            print(xkernel)
            print(ykernel)
        case _:
            print('No Suitable Mode')
            sys.exit()
    return output

                    
  
def SelectionFilter(size,mode,img):
    kernel=np.zeros((size,size),np.float32)
    padded=pad(kernel,img)
    output=np.zeros_like(img)
    for j in range (img.shape[1]):
        for i in range(img.shape[0]):
            img_part=padded[i:i+kernel.shape[0],j:j+kernel.shape[1]]
            match(mode):
                case 'min':
                    output[i,j]=min_select(kernel,img_part)
                case 'median':
                    output[i,j]=median_select(kernel,img_part)
                case 'max':
                    output[i,j]=max_select(kernel,img_part)
                case _:
                    print('No Suitable Mode')
                    sys.exit()     
    return output
  
#Main part
op=sys.argv[1]
size=int(sys.argv[2])
src=sys.argv[5]
img=cv2.imread(src)
display_image('Original',img)
parameter_flag=sys.argv[3]
if(len(sys.argv)>6):
    output_flag=sys.argv[6]
    
match(op):
    case 'G':
        sigma=int(sys.argv[4])
        output=GaussianFilter(size,sigma,img)
        display_image('Gaussian filter',output)
    case 'E':
        dir=(sys.argv[4])        
        output=EdgeDetect(size,dir,img)
        display_image('edge detected',output)
    case 'S':
        mode=(sys.argv[4])      
        output=SelectionFilter(size,mode,img) 
        display_image('Selection Filter',output)
         
    case _:
        print('No Suitable Operation')
if (len(sys.argv)>7):
    out_file=sys.argv[7]
    cv2.imwrite(out_file,output)


