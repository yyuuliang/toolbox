import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches


"""
Perspective transformation

"""
def get_M():

    # the target photo
    # dst = np.array( ((410,121),(476,121),(409,280),(475,280) ),dtype=np.float32)
    dst = np.array( ((210,121),(277,121),(210,279),(277,279) ),dtype=np.float32)

    # source photo
    src = np.array( ((178,37),(275,94),(161,282),(286,337) ),dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    M_in = cv2.getPerspectiveTransform(dst, src)

    return M, M_in

def perspectiveTransformTest(impath):
    image = mpimg.imread(impath)
    height, width, c = image.shape


    M, M_in = get_M()
    warped = cv2.warpPerspective(image, M, (image.shape[1],image.shape[0]), flags=cv2.INTER_LINEAR)
    plt.subplot(121),plt.imshow(image),plt.title('Input')
    plt.subplot(122),plt.imshow(warped),plt.title('Output')
    plt.show()


"""
Draw bounding box on a image based on minx, miny, maxx and maxy
"""

def getsize(x1,y1,x2,y2):
    w = x2-x1
    h = y2-y1
    return w,h

def draw_BB():
    impath = 'dataset/lisa/mytraining/02068.jpg'
    im = np.array(Image.open(impath), dtype=np.uint8)
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)
    w,h = getsize(818,398,882,462)
    # Create a Rectangle patch
    rect = patches.Rectangle((818,398),w,h,linewidth=1,edgecolor='r',facecolor='none')
    # Add the patch to the Axes
    ax.add_patch(rect)
    plt.show()


if __name__ == '__main__':
    impath = '/home/yuhuang/whitebase/pythontest/stops/src.jpg'
    perspectiveTransformTest(impath)