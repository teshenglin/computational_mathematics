import numpy as np
from numpy.linalg import svd
import matplotlib.pyplot as plt
from PIL import Image

def show_images(IMG):
    h, w, c = IMG.shape
    print()
    print('image information, (h, w, c) = ', (h, w, c))
    print()
    plt.imshow(IMG)
    plt.axis('off')
    plt.show()

def compress_image_orig(IMG, k):
    img = IMG
    h, w, c = img.shape

    # split r, g and b channels
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    # compressing
    ur,sr,vr = svd(r, full_matrices=False)
    ug,sg,vg = svd(g, full_matrices=False)
    ub,sb,vb = svd(b, full_matrices=False)
    rr = (ur[:,:k] * sr[:k]) @ vr[:k,:]
    rg = (ug[:,:k] * sg[:k]) @ vg[:k,:]
    rb = (ub[:,:k] * sb[:k]) @ vb[:k,:]

    # force the value to 0-255
    rr[rr>255]=255
    rg[rg>255]=255
    rb[rb>255]=255
    rr[rr<0]=0
    rg[rg<0]=0
    rb[rb<0]=0

    # arranging
    rimg = np.zeros(img.shape)
    rimg[:,:,0] = rr
    rimg[:,:,1] = rg
    rimg[:,:,2] = rb

    # convert to unit8 image
    compressed_image = rimg.astype(np.uint8)

    print(' ')
    print('Original image, (h, w, c, h*w*c)  = ', (h, w, c, h*w*c))
    print('SVD data size, c*(h*k + k + k*w) = ', c*(h*k + k + k*w))
    print(' ')

    f, (a0, a1) = plt.subplots(1, 2)
    a0.imshow(compressed_image)
    a0.axis('off')
    a0.set(title='compressed image')
    a1.imshow(img)
    a1.axis('off')
    a1.set(title='original image')
    
    plt.show()

def patch_img(img, n, m, pch_1, pch_2):
    img_patch = np.zeros((pch_1*pch_2, n*m))
    kk = 0
    for ii in range(n):
        for jj in range(m):
            img_patch[:,kk] = img[(pch_1*ii+0):(pch_1*(ii+1)), (pch_2*jj+0):(pch_2*(jj+1))].reshape((pch_1*pch_2))
            kk = kk+1

    return img_patch

def depatch_img(img_patch, n, m, pch_1, pch_2):
    img_depatch = np.zeros((n*pch_1, m*pch_2))
    kk = 0
    for ii in range(n):
        for jj in range(m):
            img_depatch[(pch_1*ii+0):(pch_1*(ii+1)), (pch_2*jj+0):(pch_2*(jj+1))] = img_patch[:,kk].reshape((pch_1, pch_2))
            kk = kk + 1

    return img_depatch

def compress_image_patched(IMG, k, patch_1, patch_2):
    img = IMG
    h, w, c = img.shape

    # transform image into patches
    n=int(h/patch_1)
    m=int(w/patch_2)
    img = img[0:n*patch_1, 0:m*patch_2, :]

    # split r, g and b channels
    img_r = img[:,:,0]
    img_g = img[:,:,1]
    img_b = img[:,:,2]

    patch_r = patch_img(img_r, n, m, patch_1, patch_2)
    patch_g = patch_img(img_g, n, m, patch_1, patch_2)
    patch_b = patch_img(img_b, n, m, patch_1, patch_2)

    # compressing

    ur,sr,vr = svd(patch_r, full_matrices=False)
    ug,sg,vg = svd(patch_g, full_matrices=False)
    ub,sb,vb = svd(patch_b, full_matrices=False)
    rr = (ur[:,:k] * sr[:k]) @ vr[:k,:]
    rg = (ug[:,:k] * sg[:k]) @ vg[:k,:]
    rb = (ub[:,:k] * sb[:k]) @ vb[:k,:]

    # force the value to 0-255
    rr[rr>255]=255
    rg[rg>255]=255
    rb[rb>255]=255
    rr[rr<0]=0
    rg[rg<0]=0
    rb[rb<0]=0

    # depatch image
    r2 = depatch_img(rr, n, m, patch_1, patch_2)
    g2 = depatch_img(rg, n, m, patch_1, patch_2)
    b2 = depatch_img(rb, n, m, patch_1, patch_2)

    # arranging
    rimg = np.zeros(img.shape)
    rimg[:,:,0] = r2
    rimg[:,:,1] = g2
    rimg[:,:,2] = b2

    # convert to unit8 image
    compressed_image = rimg.astype(np.uint8)

    print(' ')
    print('Original image, (h, w, c, h*w*c)                                   = ', (h, w, c, h*w*c))
    print('Patched  image, (patch_1*patch_2, n*m, c, patch_1*patch_2*n*m*c)   = ', (patch_1*patch_2, n*m, c, patch_1*patch_2*n*m*c))
    print('SVD data size, c*((patch_1*patch_1)*k + k + k*(n*m))              = ', c*((patch_1*patch_1)*k + k + k*(n*m)))
    print(' ')

    f, (a0, a1) = plt.subplots(1, 2)
    a0.imshow(compressed_image)
    a0.axis('off')
    a0.set(title='compressed image')
    a1.imshow(img)
    a1.axis('off')
    a1.set(title='original image')
    plt.show()
