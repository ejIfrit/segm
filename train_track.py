from model import Model
from load_data import Datagen, plot_data
import tensorflow as tf
from util import plot_segm_map_donkey, calc_iou
import numpy as np
import sys
import glob
import donkeycar as dk
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from donkeycar.management.image_thresholding.analyse_masks import loadMasks
def mySample(allData,allSegm_map,batch_size):
    idx_d = np.random.choice(allData.shape[0], batch_size)
    #idx_m = np.random.choice(allSegm_map.shape[0], batch_size)
    if batch_size<allData.shape[0]:

        return (allData[idx_d,:,:,:]-130)*(1./70.),allSegm_map[idx_d,:,:],idx_d
    else:
        return (allData-130)*(1./70.),allSegm_map,range(allData.shape[0])
batch_size = 40
dropout = 0.7
nCrop = 45
fileStubs = ['_mask','_cones']
folderName = '/Users/edwardjackson/Documents/donkeyCar/projects/donkey_ej/donkeycar/management/image_thresholding/testImages'
data = []
dataDisp = []
segm_map = []
for fn in glob.glob(folderName+'/'+'*.jpg'):
    if not any(stub in fn for stub in fileStubs):
        print(fn)
        o,m,d = loadMasks(fn,False,nCrop = nCrop,nResize = 0.25,goGrey=True)
        #fig1 = plt.figure()
        #ax1 = fig1.add_subplot(211)
        #ax2 = fig1.add_subplot(212)
        #oorig = o
        #o = (o-130.)/70.
        #oback = (o*70.+130.)
        #ax1.imshow(oorig)
        w = o.shape[0]
        h=o.shape[1]

        x = np.linspace(0., 255., h)
        y = np.linspace(0., 255., w)
        temp, yv = np.meshgrid(x, y)
        o = np.stack((o,yv),axis=-1)
        #ax2.imshow(yv)
        #plt.show()
        oflip1 = np.fliplr(o)
        dflip1 = np.fliplr(d)
        mflip1 = np.fliplr(m)
        oflip2 = np.flipud(o)
        dflip2 = np.flipud(d)
        mflip2 = np.flipud(m)
        oflip3 = np.flipud(oflip1)
        dflip3 = np.flipud(dflip1)
        mflip3 = np.flipud(mflip1)
        data.append(o)
        segm_map.append(m)
        data.append(oflip1)
        segm_map.append(mflip1)
        data.append(oflip2)
        segm_map.append(mflip2)
        data.append(oflip3)
        segm_map.append(mflip3)
        dataDisp.append(d)
        dataDisp.append(dflip1)
        dataDisp.append(dflip2)
        dataDisp.append(dflip3)
        #ax2.imshow(oback.astype('uint8'))
        #plt.show()
allData = np.array(data)
allSegm_map = np.array(segm_map)
dataDisp = np.array(dataDisp)
#dg = Datagen('data/mnist', 'data/cifar')
if len(allData.shape)==3:
    allData = allData[:,:,:,np.newaxis]
data, segm_map,temp = mySample(allData,allSegm_map,batch_size)

model = Model(batch_size, dropout,w = allData.shape[1],h=allData.shape[2],ndims = allData.shape[3])
print(data.shape)
print(segm_map.shape)

num_iter = 1500#1500
best_loss = 1.
sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
for iter in range(num_iter):
    data_batch, segm_map_batch,temp = mySample(allData,allSegm_map,batch_size)
    train_loss, _ = sess.run([model.total_loss, model.train_step], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
    if train_loss<best_loss:
        print('new best guess- saving')
        saver.save(sess, 'my_temp_model')
        best_loss = train_loss
    if iter%5 == 0:
        data_batch, segm_map_batch,temp =mySample(allData,allSegm_map,batch_size)
        test_loss, segm_map_pred = sess.run([model.total_loss, model.h4], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
        print('iter %5i/%5i loss is %5.3f and mIOU %5.3f'%(iter, num_iter, test_loss, calc_iou(segm_map_batch, segm_map_pred)))
    #print('iteration ' + str(iter))

#Final run
data_batch, segm_map_batch,idx_d = mySample(allData,allSegm_map,batch_size)
test_loss, segm_map_pred = sess.run([model.total_loss, model.h4], feed_dict={model.image:data_batch, model.segm_map:segm_map_batch})
idx_disp = plot_segm_map_donkey(dataDisp[idx_d,:,:,:], segm_map_batch, segm_map_pred)
#with tf.Session() as sess:
#  new_saver = tf.train.import_meta_graph('my_test_model.meta')
#  new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#  test_loss_load, segm_map_pred_load = sess.run([model.total_loss, new_saver.h4], feed_dict={new_saver.image:data_batch, new_saver.segm_map:segm_map_batch})
#  plot_segm_map_donkey(dataDisp[idx_d,:,:,:], segm_map_batch, segm_map_pred_load,idx_disp)
plt.show()
