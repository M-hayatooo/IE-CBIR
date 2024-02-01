import decimal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim

matplotlib.use('AGG')

# save_images(img1, reconstract1, save_dir+"compare_images_dm1.png")
# save_images(img2, reconstract2, save_dir+"compare_images_dm2.png")

LOW_DIMENSION = 175 # 150 or 175
LOW_SHAPE = (-1, 1, 5, 7, 5) # (-1, 1, 5, 6, 5) or (-1, 1, 5, 7, 5)
IMG_SHAPE = (80, 112, 80) # (80, 96, 80) or (80, 112, 80)


def save_images(image, output, epoch, path, train, ):
    # %matplotlib inline
    decimal.getcontext().prec = 3
    plt.rcParams['figure.dpi'] = 300
    # print(plt.rcParams["figure.figsize"])
    # print(plt.rcParams["figure.dpi"])
    fig = plt.figure(figsize=(18,6))
    X, Y = 2, 8

    for i in range(8):
        imgplot = i + 1
        ax1 = fig.add_subplot(X, Y, imgplot)
        ax1.set_title("original"+str(imgplot), fontsize=12)
        ax1.axis("off")
        img = np.flip(image[i].numpy().reshape(IMG_SHAPE).transpose(1,2,0)[50],0)
        plt.imshow(img, cmap="gray")
        plt.tick_params(labelsize=8)
        
        ax2 = fig.add_subplot(X, Y, imgplot+Y)
        ax2.axis("off")
        ax2.set_title("output"+str(imgplot), fontsize=12)
        out = np.flip(output[i].numpy().reshape(IMG_SHAPE).transpose(1,2,0)[50],0)
        plt.imshow(out, cmap="gray")
        ax_pos = ax2.get_position()
        mse_value = decimal.Decimal(np.sqrt(mean_squared_error(img, out)))
        mse_value += decimal.Decimal(0)
        ssim_value = decimal.Decimal(ssim(img, out, data_range=out.max() - out.min()))
        ssim_value += decimal.Decimal(0)
        fig.text(ax_pos.x1 - 0.065, ax_pos.y1 - 0.32, "rmse: " + str(mse_value), size=12)
        fig.text(ax_pos.x1 - 0.065, ax_pos.y1 - 0.365, "ssim: " + str(ssim_value), size=12)
        plt.tick_params(labelsize=8)
        # plt.tight_layout()

    if train:
        savename = f"imgs/train_rec_pic_epoch{epoch}.jpg"
    elif train is False:
        savename = f"val_imgs/val_rec_pic_epoch{epoch}.jpg"
        
    plt.savefig(path + savename)
    plt.close()
