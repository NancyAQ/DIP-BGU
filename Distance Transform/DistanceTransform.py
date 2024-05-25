import sys
import cv2
import numpy as np
def load_display(src):
    img=cv2.imread(src)
    cv2.imshow(src,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img
    
def to_gray(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def apply_thresh(img):
    val,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    return img

def remove_components(thresh_size,img):
    output_img=img.copy()
    data=cv2.connectedComponentsWithStats(output_img)
    for component in range(1,np.max(data[1])+1):
     size_to_check=data[2][component,cv2.CC_STAT_AREA]
     if (size_to_check<thresh_size): 
         output_img[data[1]==component]=0
    return output_img


def legal_point(x,y,img):
    return((x>=0 and y>=0)and(x<img.shape[0] and y<img.shape[1]))

def calculate_dist(init_map): #forwrd and backward scans
    neighbours=[(0,1),(1,0),(-1,0),(0,-1),(1,1),(-1,1),(-1,-1),(1,-1)]
    for x in range(init_map.shape[0]):
        for y in range(init_map.shape[1]):
          if(init_map[x,y]!=0): #added
            for neighbour in neighbours:
                x1=x+neighbour[0]
                y1=y+neighbour[1]
                if legal_point(x1,y1,img):
                    init_map[x,y]=min(init_map[x,y],init_map[x1,y1]+(np.sqrt(pow(x-x1,2)+pow(y-y1,2))))          
    for x in range(init_map.shape[0]-1,-1,-1):
        for y in range(init_map.shape[1]-1,-1,-1):
          if(init_map[x,y]!=0): #added
            for neighbour in neighbours:
                x1=x+neighbour[0]
                y1=y+neighbour[1]
                if legal_point(x1,y1,img):
                    init_map[x,y]=min(init_map[x,y],init_map[x1,y1]+(np.sqrt(pow(x-x1,2)+pow(y-y1,2))))
    return init_map  

def init_inner(img):
    init_map=np.ones_like(img)*np.inf 
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            if(img[i,j]==0):
                init_map[i,j]=0
    return init_map

def inner_d(img):
    init_map=init_inner(img)
    return(calculate_dist(init_map))
    
        
def init_outer(img):
    init_map=np.ones_like(img)*np.inf 
    for i in range (img.shape[0]):
        for j in range (img.shape[1]):
            if(img[i,j]==0 ):
                init_map[i,j]=0
    return init_map

def outer_d(img):
    init_map=init_outer(img)
    return(calculate_dist(init_map))


    
    
def signed_d(img):
    inverted=cv2.bitwise_not(img)
    return inner_d(img)-outer_d(inverted)
    
def normalize_display(img,file):
    normalized=cv2.normalize(img, img, 0, 1.0, cv2.NORM_MINMAX)
    if(len(sys.argv)>4): #theres a file
        cv2.imwrite(file,(normalized*255))
    else:
        cv2.imshow('Normalized', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


#main( i put comments to show the steps i made to match the assignment file :))
src=sys.argv[1]
size=int(sys.argv[2])
d_type=sys.argv[3]
file='0'
if(len(sys.argv)>4):
    file=sys.argv[4]
#1
img=load_display(src)
#2
gray_img=to_gray(img)
#3
binary_img=apply_thresh(gray_img)
#4
final=remove_components(size,binary_img)
#5
computed_dist=final.copy()
match(d_type):
    case 'I':
        computed_dist=inner_d(final)
    case 'O':
        final=cv2.bitwise_not(final)
        computed_dist=outer_d(final)
    case 'S':
        computed_dist=signed_d(final)
    case _:
        print('No Suitable Input')

#6
normalize_display(computed_dist,file)



