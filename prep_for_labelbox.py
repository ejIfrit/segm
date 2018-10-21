#!/usr/bin/env python3
"""
Look at a file and see if a mask exists for it. Analyse the image based on the mask

Usage:
    test.py (prep) [--fileName=<fn>] [--nCrop=<n1>] [--nTubs=<nt>]

Options:
    -h --help        Show this screen.
"""


def prep_for_labelBox(path,nResize,tubs='all'):
    print(path)
    print(tubs)
    print(nResize)
    pass



if __name__ == '__main__':
    args = docopt(__doc__)
    print(args)
    cd = True if args['--cd'] else False
    hsv = True if args['--hsv'] else False
    if args['--nCrop']:
        tubs =
    else:
        nc = 0
    if args['--fileName']:
        analyseMasks(args['--fileName'],isCone = cd)
    if args['--fileName'] and args['--nCrop']:
        analyseMasks(args['--fileName'],int(args['--nCrop']),isCone = cd)
    if args['folder']:# and args['--folderName']:
        analyseFolder(isCone = cd, doHSV = hsv)
    if args['clf'] and args['--test']:
        testClfMask(isCone=cd,nCrop=nc)
    if args['clf'] and not args['--test']:# and args['--folderName']:
        sm = True if args['--s'] else False
        doClf(isCone = cd, doHSV = hsv,saveModel = sm)
