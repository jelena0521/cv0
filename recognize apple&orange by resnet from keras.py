import os
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt


def load_data(path,shuffleflag = True):  #载入数据
    '''递归读取文件。只支持一级。返回文件名、数值标签、数值对应的标签名'''
    print ('loading sample  dataset..')
    filenames_path = []
    labelsnames = []
    for (dirpath, dirnames, filenames) in os.walk(path):#递归遍历文件夹
        for filename in filenames:                            #遍历所有文件名
            filename_path = os.sep.join([dirpath, filename])
            filenames_path.append(filename_path)               #添加文件名
            labelsnames.append(dirpath.split('/')[-1])     #添加文件名对应的标签 取子目录的名字

    lab= list(sorted(set(labelsnames)))  #生成标签名称列表 [trainA,trainB]
    labdict=dict(zip(lab,list(range(len(lab))))) #生成字典 {trainA:0,train:1}

    labels = [labdict[i] for i in labelsnames]  #将trainA转为0，trainB转为1
    if shuffleflag == True:
        return shuffle(np.asarray(filenames_path),np.asarray(labels))
    else:
        return (np.asarray(filenames_path),np.asarray(labels))


def distorted_image(image,size,ch=1,shuffleflag = False,cropflag = False,brightnessflag=False,contrastflag=False):    #定义函数，实现变化图片
    distorted_image =tf.image.random_flip_left_right(image)  #左右翻转

    if cropflag == True:    #随机裁剪
        s = tf.random_uniform((1,2),int(size[0]*0.8),size[0],tf.int32)  #随机生成矩阵,shape=1*2,值在size[0]*0.8-size[0]
        distorted_image = tf.random_crop(distorted_image, [s[0][0],s[0][1],ch])

    distorted_image = tf.image.random_flip_up_down(distorted_image)#上下随机翻转
    if brightnessflag == True:#随机变化亮度
        distorted_image = tf.image.random_brightness(distorted_image,max_delta=10)
    if contrastflag == True:   #随机变化对比度
        distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
    if shuffleflag==True:
        distorted_image = tf.random_shuffle(distorted_image)#沿着第0维乱序
    return distorted_image

from skimage import transform
def random_rotated30(image,label): #定义函数实现图片随机旋转操作
    def rotated(image):           #封装好的skimage模块，来进行图片旋转30度
        y, x = np.array(image.shape[:2],np.float32) / 2.
        tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(30)) #平移加旋转
        tf_shift = transform.SimilarityTransform(translation=[-x, -y])
        tf_shift_inv = transform.SimilarityTransform(translation=[x, y])
        image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
        return image_rotated

    def rotatedwrap():
        image_rotated = tf.py_func(rotated,[image],[tf.float64])   #调用第三方函数
        return tf.cast(image_rotated,tf.float32)[0]

    a = tf.random_uniform([1],0,2,tf.int32)#实现随机功能
    image_decoded = tf.cond(tf.equal(tf.constant(0),a[0]),lambda:image,rotatedwrap) #tf.cond类似于if else

    return image_decoded, label

def norm_image(image,size,ch=1,flattenflag = False):    #定义函数，实现归一化，并且拍平
    image_decoded = image/127.5-1  #image/255.0 两种都是图片归一化，前面更稀疏一点
    if flattenflag==True:
        image_decoded = tf.reshape(image_decoded, [size[0]*size[1]*ch]) #拉平
    return image_decoded

def dataset(path,size,batchsize,random_rotated=False,shuffleflag = True):#定义函数，创建数据集
    """ parse  dataset."""
    (filenames,labels),_ =load_data(path,shuffleflag=False) #载入文件名称与标签
    #print(filenames,labels)
    def parseone(filename, label):                         #解析一个图片文件
        """ Reading and handle  image"""
        image_string = tf.io.read_file(filename)         #读取整个文件
        image_decoded = tf.image.decode_image(image_string)
        image_decoded.set_shape([None, None, None])    # 必须有这句，不然下面会转化失败
        image_decoded = distorted_image(image_decoded,size)#对图片做扭曲变化
        image_decoded = tf.image.resize_images(image_decoded, size)  #变化尺寸
        image_decoded = norm_image(image_decoded,size)#归一化
        image_decoded = tf.to_float(image_decoded)
        label = tf.cast(tf.reshape(label, [1,]) ,tf.int32)#将label 转为1维度张量
        return image_decoded, label

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))#生成Dataset对象


    if shuffleflag == True:#乱序
        dataset = dataset.shuffle(10000)

    dataset = dataset.map(parseone)   #有图片内容的数据集

    if random_rotated == True:#旋转
        dataset = dataset.map(random_rotated30)

    dataset = dataset.batch(batchsize) #批次划分数据集
    dataset = dataset.prefetch(1)

    return dataset

def showresult(subplot,title,thisimg):          #显示单个图片
    p =plt.subplot(subplot)
    p.axis('off')
    p.imshow(thisimg)
    p.set_title(title)

def showimg(index,label,img,ntop):   #显示
    plt.figure(figsize=(20,10))     #定义显示图片的宽、高
    plt.axis('off')
    ntop = min(ntop,9)
    print(index)
    for i in range (ntop):
        showresult(100+10*ntop+1+i,label[i],img[i]) #几行几列当前位置
    plt.show()

from tensorflow.python.keras.applications.resnet50 import ResNet50

size = [224,224]
batchsize = 10

sample_dir="apple2orange/train"
testsample_dir = "apple2orange/test"

traindataset = dataset(sample_dir,size,batchsize)#训练集
testdataset = dataset(testsample_dir,size,batchsize,shuffleflag = False)#测试集

print(traindataset.output_types)  #打印数据集的输出信息
print(traindataset.output_shapes)


def imgs_input_fn(dataset):
    iterator = dataset.make_one_shot_iterator()			#生成一个迭代器
    one_element = iterator.get_next()					#从iterator里取出一个元素
    return one_element

next_batch_train = imgs_input_fn(traindataset)				#从traindataset里取出一个元素
next_batch_test = imgs_input_fn(testdataset)				#从testdataset里取出一个元素

#sess=tf.Session()	# 建立会话（session）
#sess.run(tf.global_variables_initializer())  #初始化
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        for step in np.arange(1):
            value = sess.run(next_batch_train)
            showimg(step,value[1],np.asarray((value[0]+1)*127.5,np.uint8),10)       #显示图片

    except tf.errors.OutOfRangeError:           #捕获异常
        print("Done!!!")


###########################################
#构建模型
img_size = (224, 224, 3)
inputs = tf.keras.Input(shape=img_size)
conv_base = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',input_tensor=inputs,input_shape = img_size,include_top=False)#创建ResNet网络

model = tf.keras.models.Sequential()
model.add(conv_base)
model.add(t)
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
conv_base.trainable = False
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])
#训练模型
model_dir ="./models/app2org"
os.makedirs(model_dir, exist_ok=True)
print("model_dir: ",model_dir)
est_app2org = tf.keras.estimator.model_to_estimator(keras_model=model,  model_dir=model_dir)

#训练模型
#传入数据
train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(traindataset),max_steps=500)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(testdataset))

import time
start_time = time.time()
tf.estimator.train_and_evaluate(est_app2org, train_spec, eval_spec) #训练和评估模型
print("--- %s seconds ---" % (time.time() - start_time))

#测试模型
img = value[0]
lab = value[1]

pre_input_fn = tf.estimator.inputs.numpy_input_fn(img,batch_size=10,shuffle=False)
predict_results = est_app2org.predict(input_fn=pre_input_fn)

predict_logits = []
for prediction in predict_results:
    #print(prediction)
    predict_logits.append(prediction['dense_1'][0])

predict_is_org = [int(np.round(logit)) for logit in predict_logits]
actual_is_org = [int(np.round(label[0]))  for label in lab]
showimg(step,value[1],np.asarray( (value[0]+1)*127.5,np.uint8),10)
print("Predict :",predict_is_org)
print("Actual  :",actual_is_org)


