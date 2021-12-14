import numpy as np
import cv2


def rle2mask(mask_rle, shape=(2100, 1400)):
    '''
    For the Kaggle image dataset
    Change the 1-line Encoding Pixels to a 2D mask
    mask_rle: run-length as string formated (start length)
    shape: (width,height) of array to return
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def extract_contour(img, cnts, shown=False, direct=True):
    '''
        For the Kaggle image dataset
        Extract from contours images
    '''
    #cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rois = []
    # Find bounding box and extract ROI
    #print(cnts)
    for c in cnts:
        # The cv2.boundingRect() function of OpenCV is used to draw 
        # an approximate rectangle around the binary image.
        x,y,w,h = cv2.boundingRect(c)
        ROI = img[y:y+h, x:x+w]
        if direct:
            return ROI
        rois.append(ROI)
    print('ROI: ',rois)
    if not shown:
        plt.figure()
        plt.imshow(ROI)
    return rois


