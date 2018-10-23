#!/usr/bin/env python3
"""
Input a list of tubs to transfer, put them in a folder at 4x size to make them
easier to label in labelbox
nTubs variable is a vector where you put in a seris of 'start:stop:strides', e.g.
--nTubs [1:10:2,20:30:3] would give tub numbers 1,3,5,7,9,20,23,26,29
path is a folder with donkeycar tubs in it
dirtosave is the directory name that the images get saved to 
Usage:
    test.py (prep) [--path=<fn>] [--nResize=<n1>] [--nTubs=<nt>]

Options:
    -h --help        Show this screen.
"""
import os
import sys
from docopt import docopt
import glob
import ast
import numpy as np
import cv2
def prep_for_labelBox(path,nResize,tubs='all',dirToSave = 'forLabelling'):
    print(path)
    print(tubs)
    print(nResize)
    tubsList = []
    if tubs!='all':
        tubsList = eval('np.r_'+tubs)
        print(tubsList)
        #print(fn[len(path)+1:])
    if not os.path.exists(path+'/'+dirToSave):
        os.makedirs(path+'/'+dirToSave)

    for fn in glob.glob(path+'/'+'*.jpg'):
            tubNo = int("".join(filter(str.isdigit, fn[len(path)+1:])))
            if tubs!='all' and tubNo in tubsList:
                print(tubNo)
                img1_orig = cv2.imread((fn))
                img1_out =img1_orig#= cv2.cvtColor(img1_orig, cv2.COLOR_BGR2RGB)
                if nResize!=1:
                    height,width,depth = img1_out.shape
                    img1_out = cv2.resize(img1_out,(int(width*nResize),int(height*nResize)))
                    cv2.imwrite(os.path.join(path+'/'+dirToSave , str(tubNo)+'resized'+'.jpg'), img1_out)


if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)

    tubsIn = args['--nTubs'] if args['--nTubs'] else 'all'
    if args['--path'] and args['--nResize'] and args['--nTubs']:
        prep_for_labelBox(args['--path'],nResize = int(args['--nResize']),tubs=tubsIn)
