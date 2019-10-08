# 仅作为学习日记，不可用于任何商业用途
#当样本有限时，最好的办法就是微调模型
#1、定义读取图片以及label（label需转为数字）
#2、定义对图片的变化，亮度、对比度、左右旋转、上下旋转，剪切
#3、定义对图片指定角度的旋转
#4、定义对图片归一化 image/255以及image/127.5-1
#5、根据1，利用tf.data.Dataset.from_tensor_slices生成dataset
#6、利用dataset.map实现2、3、4的转变
#7、利用dataset.batch实现批次
#8、指定train和test的dir 分别获得他们的dataset
#9、定义model，model需加上flatten层、两个全连接层
#10、model.compile加上损失、学习率、acc
#11、定义model存入的dir，实例化模型
#12、将train和test的数据代入tf.estimator.train_and_evaluate获得准确率


