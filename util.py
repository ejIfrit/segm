import numpy as np
import matplotlib.pyplot as plt

def plot_segm_map(data, segm_map, segm_map_pred):
    num_samples = data.shape[0]

    num_plot = 5

    f,ax = plt.subplots(num_plot, 3)
    f.tight_layout()

    for row in range(num_plot):
        idx = np.random.randint(0,num_samples)
        datashow = data[idx]*70+130

        ax[row, 0].imshow(datashow.astype('uint8'))
        ax[row, 1].imshow(segm_map[idx])
        ax[row, 2].imshow(segm_map_pred[idx])
        for axis in ax[row]:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        if row==0:
            ax[0,0].set_title('Input image')
            ax[0,1].set_title('Target segmentation map')
            ax[0,2].set_title('Pred segmentation map')
    f.subplots_adjust(hspace=0.1)  #No horizontal space between subplots
    f.subplots_adjust(wspace=0)
    plt.show()

def plot_segm_map_donkey(data, segm_map, segm_map_pred,idx=None):
    num_samples = data.shape[0]

    num_plot = 5

    f,ax = plt.subplots(num_plot, 3)
    f.tight_layout()
    if idx is None:
        idx = np.random.choice(data.shape[0], num_plot)
        #print(idx)
    for row,id in zip(range(num_plot),idx):

        if data.shape[-1] == 2:
            data = data[:,:,:,0]
        datashow = np.squeeze(data[id])#*70.+130.)
        ax[row, 0].imshow(datashow.astype('uint8'))
        ax[row, 1].imshow(segm_map[id])
        ax[row, 2].imshow(segm_map_pred[id])
        for axis in ax[row]:
            axis.get_xaxis().set_visible(False)
            axis.get_yaxis().set_visible(False)
        if row==0:
            ax[0,0].set_title('Input image')
            ax[0,1].set_title('Target segmentation map')
            ax[0,2].set_title('Pred segmentation map')
    f.subplots_adjust(hspace=0.1)  #No horizontal space between subplots
    f.subplots_adjust(wspace=0)
    return idx
    #plt.show()


def calc_iou(segm_map, segm_map_pred):
    segm_map_pred = np.round(segm_map_pred)
    intersection = np.sum(np.logical_and(segm_map, segm_map_pred),axis=(1,2))
    union = np.sum(np.logical_or(segm_map, segm_map_pred),axis=(1,2))
    return np.mean(intersection/union)
