from model import Model,Model_fromLoad
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


print(data.shape)
print(segm_map.shape)

#Final run
data_batch, segm_map_batch,idx_d = mySample(allData,allSegm_map,batch_size)

#with tf.Session() as sess:
  #sess.run(tf.global_variables_initializer())
#

#print(model.h4)
sess = tf.Session()
#
sess.run(tf.global_variables_initializer())
graph = tf.get_default_graph()

#print(sess.run(graph.get_tensor_by_name("bias1:0")))
new_saver = tf.train.import_meta_graph('my_test_model_1500.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
#b1 = tf.get_variable('bias1',[10,])
model = Model_fromLoad(graph,batch_size,dropout,w = allData.shape[1],h=allData.shape[2],ndims = allData.shape[3])
print(allData.shape)
#segm_map_pred_load = sess.run(model.h4, feed_dict={model.image:data_batch})
#a = graph.get_tensor_by_name("bias1:0")
#tf.Print(a, [a], message="This is a: ")
# Add more elements of the graph using a
print(sess.run(graph.get_tensor_by_name("bias1:0")))
print(sess.run(graph.get_tensor_by_name("weight1:0"))[0,0,0,:])
#print(model.h4)
segm_map_pred_load = sess.run(model.h4, feed_dict={model.image:(allData[0:40,:,:,:]-130)*(1./70.)})
plot_segm_map_donkey(dataDisp[0:5,:,:,:], segm_map_batch, segm_map_pred_load)
print('done')
plt.show()
