import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
def display(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def plot_help(hist,graph_name):
    plt.figure()
    plt.title(graph_name)
    plt.plot(hist)
    plt.bar(np.arange(len(hist)),hist)
    plt.xlabel('intensity')
    plt.ylabel('nk/freq')
    plt.show()

def accumulator(Hist):
    acc=[Hist[0]]
    for val in Hist[1:]:
        acc.append(val+acc[-1])
    return np.array(acc)


def CalcHistogram(img):
    hist=[0]*256
    img_array=np.asarray(img)
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            hist[img_array[i,j]]+=1
    return hist
           
    
    
def NormalizeHistogram(histogram,img_dim):
    dim=np.prod(img_dim)
    histogram=histogram/dim
    return histogram
    
    
def EqualizeHistogram(histogram,img):
    accum=accumulator(histogram) 
    tranformation = np.floor(255 * accum).astype(np.uint8) #255 is the L-1 we talked about in the lecture
    img1=np.asarray(img)
    img1=list(img1.flatten())
    equalized=[tranformation[pix] for pix in img1]
    equalized_img=np.reshape(np.asarray(equalized),img.shape)
    display('Equalized image',equalized_img)
    plot_help(NormalizeHistogram(CalcHistogram(equalized_img),equalized_img.shape),'Equalized Histogram')
    return equalized_img

def MatchHistogram(img1,img2):
    cdf1=accumulator(NormalizeHistogram(CalcHistogram(img1),img1.shape))
    cdf2=accumulator(NormalizeHistogram(CalcHistogram(img2),img2.shape))
    map=np.arange(256)
    matched_pix=np.interp(cdf2,cdf1,map)
    matched=np.interp(img2,map,matched_pix).astype(np.uint8)
    matched=np.reshape(matched,img2.shape)
    display('matched',matched)
    plot_help(NormalizeHistogram(CalcHistogram(matched),matched.shape),'Matched Histogram')
    return matched
    


#MAIN PART  

op=sys.argv[1]
src=sys.argv[2]
if(len(sys.argv)>3 and op=='M'):
    match=sys.argv[3]
    if(len(sys.argv)>4):
        flag=sys.argv[4]
        output=sys.argv[5]
img=cv2.imread(src,0) #reading image in gray scale
display('input image',img) 
if(len(sys.argv)>3 and op=='M'):
    img2=cv2.imread(match,0)
    display('image to match',img2)

match(op):
    case 'H':
        hist=CalcHistogram(img)
        hist=NormalizeHistogram(hist,img.shape)
        plot_help(hist,'calculated histogram')
    case 'E':
        Histogram=CalcHistogram(img)
        Equalized=EqualizeHistogram(NormalizeHistogram(Histogram,img.shape),img)
        if(len(sys.argv)>3):
            flag=sys.argv[3]
            output=sys.argv[4]
            cv2.imwrite(output,Equalized)
        
    case 'M':
        img1=img
        matched=MatchHistogram(img1,img2)
        if(len(sys.argv)>4):
            cv2.imwrite(output,matched)
        
    case _:
        print('No Suitable Operation')
        
