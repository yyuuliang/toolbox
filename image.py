import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import os

import imageio
import imgaug as ia
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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

"""
Image Augumention
"""

def augument(impath):
    ia.seed(1)

    image = imageio.imread(impath)

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=170,y1=130,x2=252,y2=248,label='18'),
        BoundingBox(x1=100,y1=154,x2=120,y2=175,label='1')

    ], shape=image.shape)

    ia.imshow(bbs.draw_on_image(image, size=2))

    # apply augumentation
    #  We choose a simple contrast augmentation (affects only the image) and an affine transformation (affects image and bounding boxes).
    seq = iaa.Sequential([
        iaa.GammaContrast(1.5),
        iaa.Affine(translate_percent={"x": 0.1}, scale=0.8)
    ])

    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    ia.imshow(bbs_aug.draw_on_image(image_aug, size=2))

    #  we apply an affine transformation consisting only of rotation.
    image_aug, bbs_aug = iaa.Affine(rotate=50)(image=image, bounding_boxes=bbs)
    ia.imshow(bbs_aug.draw_on_image(image_aug))


def aug_resize(data_path):
    num_total = 8877
    imglist = ['00800.jpg', '04000.jpg','08000.jpg']
    idx = 0

    for i in range(3):
        fname = imglist[i]
        imgpath = os.path.join(data_path, fname)
        print(imgpath)
        # we have to resize all the images to same size, along with the bounding boxes
    print('all: ', idx)


if __name__ == '__main__':
    cwd = os.getcwd()
    # impath = os.path.join(cwd,'stops/src.jpg')
    # perspectiveTransformTest(impath)
    # augument(impath)

    data_path = 'tfrecord-images/training_data'
    aug_resize(data_path)

    # imgpath = '/home/yuhuang/singitlab/cameraprojects/traffic-sign-recognition/test-images/04644.jpg'
    # crop(imgpath)