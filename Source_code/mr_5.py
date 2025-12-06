from PIL import Image, ImageOps
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import Image
class MyGaussianBlur(ImageFilter.Filter):
    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur((self.radius, self.radius))
            image.paste(clips, self.bounds)
            return image
        else:

            return image.gaussian_blur((self.radius, self.radius))


def mr_brightness(img, p):
    enh_bri = ImageEnhance.Brightness(img)
    brightness = p
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

# MR1: Flip left-right
def test_flip_left_right(image):
    flipped_image = ImageOps.mirror(image)
    return flipped_image

# MR2: Flip up-down
def test_flip_up_down(image):
    flipped_image = ImageOps.flip(image)
    return flipped_image

def mr_gaussian(img,p):
    radius_num = 2
    new_im = img.filter(MyGaussianBlur(radius=radius_num))
    return new_im

def mr_colored(img, p):
    enh_col = ImageEnhance.Color(img)
    color = p
    im4 = enh_col.enhance(color)
    return im4

# MR3: Rotate 5º
def test_rotate(image, angle):
    rotated_image = image.rotate(angle)
    return rotated_image

# MR5: Shear
def test_shear(image, shear_factor):
    width, height = image.size
    xshift = abs(shear_factor) * width
    new_width = width + int(round(xshift))
    image = image.transform((new_width, height), Image.AFFINE,
                            (1, shear_factor, -xshift if shear_factor > 0 else 0, 0, 1, 0), Image.BICUBIC)
    return image

# 测试所有 MR
def test_mrs(original_image, mr_p):
    global a
    mr = mr_p.split()[0]
    p = float(mr_p.split()[1])
    if mr == 'flip_left_right':
        a = test_flip_left_right(original_image)
    elif mr == 'rotate' and p == 5:
        a = test_rotate(original_image, 5)
    elif mr == 'brightness':
        a = mr_brightness(original_image, p)
    elif mr == 'gaussian':
        a = mr_gaussian(original_image, p)
    elif mr == 'colored':
        a = mr_colored(original_image, p)
    return a



