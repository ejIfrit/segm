from model import Model
from load_data import Datagen, plot_data
import tensorflow as tf
from util import plot_segm_map_donkey, calc_iou
import numpy as np
import sys
import glob
import donkeycar as dk
from donkeycar.management.image_thresholding.analyse_masks import loadMasks
def mySample(allData,allSegm_map,batch_size):
    idx_d = np.random.choice(allData.shape[0], batch_size)
    idx_m = np.random.choice(allSegm_map.shape[0], batch_size)
    return allData[idx_d,:,:,:],allSegm_map[idx_m,:,:]

batch_size = 5
dropout = 0.7
nCrop = 45
fileStubs = ['_mask','_cones']
folderName = '/Users/edwardjackson/Documents/donkeyCar/projects/donkey_ej/donkeycar/management/image_thresholding/testImages'
data = []
segm_map = []
for fn in glob.glob(folderName+'/'+'*.jpg'):
    if not any(stub in fn for stub in fileStubs):
        print(fn)
        o,m = loadMasks(fn,False)
        if nCrop>0:
            o = o[nCrop:,:,:]
            m = m[nCrop:,:]

        data.append(o)
        segm_map.append(m)
allData = np.array(data)
allSegm_map = np.array(segm_map)
#dg = Datagen('data/mnist', 'data/cifar')
data, segm_map = mySample(allData,allSegm_map,batch_size)

model = Model(batch_size, dropout,w = allData.shape[1],h=allData.shape[2])
print(data.shape)
print(segm_map.shape)

num_iter = 500

sess = tf.Session()

sess.run(tf.global_variables_initializer())
for iter in range(num_iter):
    data_batch, segm_map_batch = mySample(allData,allSegm_map,batch_size)
    train_loss, _ = sess.run([model.total_loss, model.train_step], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})

    if iter%5 == 0:
        data_batch, segm_map_batch =mySample(allData,allSegm_map,batch_size)
        test_loss, segm_map_pred = sess.run([model.total_loss, model.h4], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
        print('iter %5i/%5i loss is %5.3f and mIOU %5.3f'%(iter, num_iter, test_loss, calc_iou(segm_map_batch, segm_map_pred)))
    #print('iteration ' + str(iter))

#Final run
data_batch, segm_map_batch = mySample(allData,allSegm_map,batch_size)
test_loss, segm_map_pred = sess.run([model.total_loss, model.h4], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
plot_segm_map_donkey(data_batch, segm_map_batch, segm_map_pred)
