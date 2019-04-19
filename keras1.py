# 首先加载keras自带的mnist数据集
from keras.datasets import mnist
# 自带的mnist第一次需要下载，返回两个元组对象，每个元组里是两个numpy数组，我们把它拆分成四个numpy数组数据集
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
# 我们可以看看他们的形状
print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
#首先我们导入keras的神经网络模型包models和层包layer
from keras import models
from keras import layers
#定义一个网络的模型对象firstnet
firstnet=models.Sequential()
#Sequential模型代表我们的模型网络是按层顺序叠加的
#为网络添加第一层
#创建一个Dense（全联接层） 使用激活函数为relu函数，输入n个28*28维的向量（1个2D张量），激活后输出n个512维的向量
firstnet.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
#第二层 softmax层
firstnet.add(layers.Dense(10,activation='softmax'))
#编译阶段 使用rmsprop优化器，交叉熵作为损失函数，监控指标为准确度accuracy
firstnet.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
#数据处理 
#我们的数据格式与编写的网络不同，网络要求为n个28*28维的向量（2D张量），我们的是3D张量（60000，28，28）需要修改成（60000，28*28）
#并且将[0,255]的灰度范围映射到[0,1],并把数据类型由uint更改为float32
train_images=train_images.reshape(60000,28*28)
train_images=train_images.astype('float32')/255
test_images=test_images.reshape(10000,28*28)
test_images=test_images.astype('float32')/255
#我们还需要对标签进行分类编码，未来将会对这一步骤进行解释。
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
#现在我们准备开始训练网络，在 Keras 中这一步是通过调用网络的 fit 方法来完成的—— 我们在训练数据上拟合(fit)模型。
firstnet.fit(train_images,train_labels,epochs=5,batch_size=128)
#训练过程中显示了两个数字:一个是网络在训练数据上的损失(loss)，另一个是网络在 训练数据上的精度(acc)。
#我们很快就在训练数据上达到了 0.989(98.9%)的精度。现在我们来检查一下模型在测试 集上的性能。
test_loss,test_acc=firstnet.evaluate(test_images,test_labels)
print('test_loss: ',test_loss,'test_acc: ',test_acc)



