from random import randint
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from image_base import Image

def predict(img, path_to_model='./unet/model.pt',device='cpu'):
    model = torch.load(path_to_model, map_location=device)
    model.eval()
    result = model(torch.tensor(img.image).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float))
    return result

def find_nearest(img, path_to_model='./unet/model.pt', path_to_collection='../dataset/', device='cpu'):
    # cur_im = Image('../test_corr.png')
    our_predict = predict(img , path_to_model, device)
    collection = os.listdir(path_to_collection)
    min_index = -1
    min_value = 1e10
    for j, item in enumerate(collection):
        img_processed = Image(os.path.join(path_to_collection, item))
        dist = mse(our_predict[0], img_processed.image)
        if dist < min_value:
            min_value = dist
            min_index = j

    result = Image(os.path.join(path_to_collection, collection[min_index]))
    return result

def mse(imageA, imageB):
    imageA = imageA.detach().numpy()
    err = np.sum((imageA - imageB) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
	
    return err

def bilinear_interpolate(image, x, y, channels=1):
    """
    """
    height, width = image.shape
    # Calculate the coordinates of the four surrounding pixels
    x1, y1 = int(x), int(y)
    x2, y2 = min(x1 + 1, width - 1), min(y1 + 1, height - 1)
    # Calculate the differences
    dx, dy = x - x1, y - y1
    # Interpolate
    interpolated = (image[y1, x1] * (1 - dx) * (1 - dy) +
                        image[y1, x2] * dx * (1 - dy) +
                        image[y2, x1] * (1 - dx) * dy +
                        image[y2, x2] * dx * dy)
    return interpolated

def create_patches(image, size=9) :
    """
    """
    height, width = image.shape
    patches = []
    patche_size = size
    for i in range(10000) :
        posx = randint(0, height - patche_size)
        posy = randint(0, width - patche_size)
        patches.append(image[posx:posx + patche_size, posy:posy+patche_size])
    return patches, patche_size

def create_mask(image, *args) :
    """
    """
    # Usage : create_mask([5,15], [[25,50], [25,50]])
    restoration_mask = np.zeros_like(image, dtype=bool)
    for xcoords, ycoords in args :
        try :
            len(xcoords)
        except :
            xcoords = [xcoords, xcoords+1]
        try :
            len(ycoords)
        except :
            ycoords = [ycoords, ycoords+1]
        xcoords.sort()
        ycoords.sort()
        restoration_mask[xcoords[0]:xcoords[1], ycoords[0]:ycoords[1]] = True
    return restoration_mask

def coeffs(image, center, xset, yset, a = 0.1, b = 0.5, fading_range = 0.1, angle = 0.) :
    """
    """
    coeffslist = []
    # transforms the given arguments to have a list of all x and y for which coefficiens are needed
    try :
        len(xset)
    except :
        xset = np.array([xset])
    else :
        xset = list(range(*xset))
    try :
        len(yset)
    except :
        yset = np.array([yset])
    else :
        yset = list(range(*yset))

    for i in xset :
        tmp = []
        for j in yset :
            direction = np.array([i, j])-np.array([center[0], center[1]])
            r = np.linalg.norm(direction)
            beta = 0 if r==0 else np.arctan2(j-center[1], i-center[0]) # pi/2 rotation of the direction, so that the angle 0 corresponds to the vertical axis
            shift =	min(b+np.cos(beta-angle)**2*fading_range, 1)*max(image.shape)/2 # The max allowed distance is half the image height
            tmp.append(1/2 - np.arctan(a*(r-shift))/np.pi)
        coeffslist.append(tmp)
    return np.array(coeffslist)


def pad_array(cur_arr, pad_width, mode='constant'):
    """
    Function that allows you to add a padding to a matrix with 
    a certain width and select one of three variants of its filling.

    :param cur_arr: input matrix that needs padding
    :type cur_arr: np.array
    :param pad_width: padding width to be applied to each side
    :type pad_width: int
    :param mode: style of filling new matrix cells, can be ['constant', 'symmetric', 'edge']
    :type mode: str

    :return: padded input matrix
    :rtype: np.array
    """
    if pad_width > cur_arr.shape[0]:
        raise ValueError("This implementation supports only padding smaller than image width.")

    h, w = cur_arr.shape
    padded_arr = np.zeros((h + 2 * pad_width, w + 2 * pad_width))

    padded_arr[pad_width : h + pad_width, pad_width : w + pad_width] = cur_arr

    if mode == 'symmetric':
        padded_arr[:pad_width, pad_width : w + pad_width] = np.flip(cur_arr[:pad_width, :], axis=0)
        padded_arr[h + pad_width:, pad_width : w + pad_width] = np.flip(cur_arr[h - pad_width:, :], axis=0)
        padded_arr[:, :pad_width] = np.flip(padded_arr[:, pad_width : 2 * pad_width], axis=1)
        padded_arr[:, w + pad_width:] = np.flip(padded_arr[:, w : w + pad_width], axis=1)

    elif mode == 'edge':
        padded_arr[:pad_width, :] = padded_arr[pad_width, :]
        padded_arr[h + pad_width:, :] = padded_arr[h + pad_width - 1, :]
        padded_arr[:, :pad_width] = padded_arr[:, pad_width:pad_width + 1]
        padded_arr[:, w + pad_width:] = padded_arr[:, w + pad_width - 1 : w + pad_width]

    return padded_arr

# if __name__ == "__main__":
#     pr_im = find_nearest()
#     print(pr_im)
#     plt.imshow(pr_im.image, cmap='gray')
#     # plt.imshow(pr_im.image.permute(1, 2, 0).detach().numpy(), cmap='gray')
#     plt.show()