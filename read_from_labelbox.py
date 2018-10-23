import sys
import json
import glob
from skimage import io
import matplotlib as mpl
import cv2
import numpy as np
mpl.use('TkAgg')
import matplotlib.pyplot as plt
def getMaskFromLabelBox(export_folder,labelKey='Track',nResize=1):
    #export_folder = sys.argv[1]
    export_file = glob.glob(export_folder+'/'+'*.json')
    with open(export_file[0]) as f:
        loaded_json = json.load(f)
        ims = []
        masks = []
        for x in loaded_json:
            print(x)
            url = x['Label']['segmentationMasksByName']['Track']
            print(url)
            mask = io.imread(url)
            mask = mask[:,:,:]
            gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            ret,thresh1 = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
            #plt.imshow(thresh1)
            url = x['Labeled Data']
            imOut = io.imread(url)
            #plt.imshow(thresh1)
            if nResize!=1:
                height,width,depth = imOut.shape
                imOut = cv2.resize(imOut,(int(width*nResize),int(height*nResize)))
                thresh1 = cv2.resize(thresh1,(int(width*nResize),int(height*nResize)))

            ims.append(imOut)
            masks.append(thresh1)
        ims = np.array(ims)
        masks = np.array(masks)
        im_grays = []
        for o in ims:
            img1 = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
            im_grays.append(img1)
        im_grays = np.array(im_grays)
        print(ims.shape)
        print(im_grays.shape)
        print(masks.shape)
        return im_grays,masks,ims
            #plt.imshow(imOut)
            #plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        fail_for_missing_file()

    export_folder = sys.argv[1]
    export_file = glob.glob(export_folder+'/'+'*.json')
    getMaskFromLabelBox(export_folder)
    #with open(export_file[0]) as f:

    #    loaded_json = load(f)
    #    for x in loaded_json:
        	#print(x)

            #url = x['Label']['segmentationMaskURL']
    #        url = x['Label']['segmentationMasksByName']['Track']
    #        print(url)
    #        image = io.imread(url)
    #        plt.imshow(image)
    #        plt.show()
            #print(loaded_json[x])
