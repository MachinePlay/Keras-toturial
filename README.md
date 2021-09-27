# Keras-toturial
>本文代码仓库[https://github.com/MachinePlay/Keras-toturial](https://github.com/MachinePlay/Keras-toturial) 包含.py 代码和Jupyter Notebook的.ipynb文件

# Keras
Keras是一个高度封装的python深度学习库，以TensorFlow或Thano为后端
最近打算重新系统的学习深度学习，于是就从使用Keras开始从零撸神经网络
Keras 具有以下重要特性。
- 相同的代码可以在 CPU 或 GPU 上无缝切换运行。
- 具有用户友好的 API，便于快速开发深度学习模型的原型。
- 内置支持卷积网络(用于计算机视觉)、循环网络(用于序列处理)以及二者的任意
组合。
- 支持任意网络架构:多输入或多输出模型、层共享、模型共享等。这也就是说，Keras
- 能够构建任意深度学习模型，无论是生成式对抗网络还是神经图灵机。

Keras 已有 200 000 多个用户，既包括创业公司和大公司的学术研究人员和工程师，也包括 研究生和业余爱好者。Google、Netflix、Uber、CERN、Yelp、Square 以及上百家创业公司都在 用 Keras 解决各种各样的问题。Keras 还是机器学习竞赛网站 Kaggle 上的热门框架，最新的深度学习竞赛中，几乎所有的优胜者用的都是 Keras 模型
# Keras、TensorFlow、Theano 和 CNTK
Keras 是一个模型级(model-level)的库，为开发深度学习模型提供了高层次的构建模块。 它不处理张量操作、求微分等低层次的运算。相反，它依赖于一个专门的、高度优化的张量库 来完成这些运算，这个张量库就是 Keras 的后端引擎(backend engine)。Keras 没有选择单个张 量库并将 Keras 实现与这个库绑定，而是以模块化的方式处理这个问题
![image.png](https://upload-images.jianshu.io/upload_images/4064394-e6b8036d077707e6.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
因此，几 个不同的后端引擎都可以无缝嵌入到 Keras 中。目前，Keras 有三个后端实现:TensorFlow 后端、 Theano 后端和微软认知工具包(CNTK，Microsoft cognitive toolkit)后端。

# 配置要求
本文使用Ubuntu18.04 搭载Nvidia GTX 1080Ti显卡，windows+cpu也行。

在开始开发深度学习应用之前，你需要建立自己的深度学习工作站。虽然并非绝对必要，但强烈推荐你在现代 NVIDIA GPU 上运行深度学习实验。某些应用，特别是卷积神经网络的图 像处理和循环神经网络的序列处理，在 CPU 上的速度非常之慢，即使是高速多核 CPU 也是如此。 即使是可以在 CPU 上运行的深度学习应用，使用现代 GPU 通常也可以将速度提高 5 倍或 10 倍。 如果你不想在计算机上安装 GPU，也可以考虑在 AWS EC2 GPU 实例或 Google 云平台上运行深 度学习实验。但请注意，时间一长，云端 GPU 实例可能会变得非常昂贵。

 无论在本地还是在云端运行，最好都使用 UNIX 工作站。虽然从技术上来说可以在 Windows 上 使 用 K e r a s ( K e r a s 的 三 个 后 端 都 支 持 W i n d o w s )， 但 我 们 不 建 议 这 么 做 。
# 环境搭建
python环境建议使用anaconda安装，anaconda安装教程可以参考我的[Anaconda简单入门](https://www.jianshu.com/p/742dc4d8f4c5)
Keras以TensorFlow为后端，所以要先安装Tensorflow
## 1.安装显卡驱动 （安装cpu版本跳至步骤2）

可以先安装显卡驱动（只安驱动就可以，不用像其他教程那样再安装cuda和cudnn）windows显卡驱动安装还请自行解决 本文主要以ubuntu 18.04讲解
- .首先卸载旧版本显卡驱动：
```
$sudo apt-get purge --remove nvidia* 命令卸载旧版本
```  
- 添加驱动源  
```
$sudo add-apt-repository ppa:graphics-drivers/ppa
$sudo apt update
```  
- 安装驱动  
首先，检测你的NVIDIA图形卡和推荐的驱动程序的模型。执行命令：
```
ubuntu-drivers devices 
```
输出结果为：
```
== /sys/devices/pci0000:00/0000:00:01.0/0000:01:00.0 ==
modalias : pci:v000010DEd00001B81sv000019DAsd00001434bc03sc00i00
vendor   : NVIDIA Corporation
model    : GP102 [GeForce GTX 1080 Ti]
driver   : nvidia-driver-396 - third-party free
driver   : nvidia-driver-390 - third-party free
driver   : nvidia-driver-415 - third-party free recommended
driver   : nvidia-driver-410 - third-party free
driver   : xserver-xorg-video-nouveau - distro free builtin

```
从中可以看到，这里有一个设备是GTX 1080 Ti ，对应的驱动是nvidia-driver-396,415,410 ，而推荐是安装415版本的驱动

```
$sudo apt install nvidia-driver-415 安最新的就行
```

## 2安装TensorFlow
anaconda创建一个新环境（或者用旧环境 例如我的环境叫tf3.6）
```
conda create -n tf3.6（名字随便起） python =3.6# 指定使用python3.6版本
```
- 如果有Nvidia显卡并装好驱动，可安装GPU版TensorFlow
在新环境（tf3.6）中
```
conda install tensorflow-gpu=1.12 #不指定版本号可以安装最新的 目前最新的是1.12
```

conda会自动把需要的numpy、scikit、cuda、cudnn装在新环境里 
- 没有Nvidia显卡 安装cpu版TensorFlow
```
conda install tensorflow
 ```
在Terminal输入
```
python
>>> import tensorflow as tf
```
如下图不报错就是Tensorflow安装成功了
![image.png](https://upload-images.jianshu.io/upload_images/4064394-b933155cee894aa0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 3 安装Keras和Jupyter notebook
 ```
conda install keras
conda install jupyter notebook
```
jupyter notebook是可以通过浏览器写python代码的一个小工具，非常好用
在终端输入
  ```
jupyter notebook
```
即可开启notebook服务，复制滚出的地址在浏览器打开，点击左上角New一个python3 notebook就可以在网页写代码和运行啦
![image.png](https://upload-images.jianshu.io/upload_images/4064394-31c987652e6d2d90.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
编辑界面
![image.png](https://upload-images.jianshu.io/upload_images/4064394-c6cbf865e5403351.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
还可以开启远程访问、终端访问等功能可以参考我的[Jupyter Notebook 远程访问配置](https://www.jianshu.com/p/3cc167bd63dd)一文
# Keras MNIST启动～
![image.png](https://upload-images.jianshu.io/upload_images/4064394-ba4178e4561159eb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们这里要解决的问题是，将手写数字的灰度图像(28 像素×28 像素)划分到 10 个类别 中(0~9)。我们将使用 MNIST 数据集，它是机器学习领域的一个经典数据集，其历史几乎和这 个领域一样长，而且已被人们深入研究。

这个数据集包含 60 000 张训练图像和 10 000 张测试图 像，由美国国家标准与技术研究院(National Institute of Standards and Technology，即 MNIST 中 的 NIST)在 20 世纪 80 年代收集得到。你可以将“解决”MNIST 问题看作深度学习的“Hello World”，正是用它来验证你的算法是否按预期运行。当你成为机器学习从业者后，会发现 MNIST 一次又一次地出现在科学论文、博客文章等中。
# 首先加载keras自带的mnist数据集
```
from keras.datasets import mnist
```
- 自带的mnist第一次需要下载，返回两个元组对象，每个元组里是两个numpy数组，我们把它拆分成四个numpy数组数据集
```
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
```
- 我们可以看看他们的形状
```
print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
```
![image.png](https://upload-images.jianshu.io/upload_images/4064394-72dfcf52fa25d02a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 接下来的工作流程
- 首先将训练数据 train_images、train_labels、测试数据test_images,test_labels转化成网络要求的数据格式（由灰度（0，255）转换为[0,1]之间的数据
- 其次设计网络模型 选择优化器、损失函数和监控指标
- 最后使用我们训练好的模型对test_image进行预测

神经网络的核心组件是层(layer)，它是一种数据处理模块，你可以将它看成数据过滤器。 进去一些数据，出来的数据变得更加有用。具体来说，层从输入数据中提取表示——我们期望 这种表示有助于解决手头的问题。大多数深度学习都是将简单的层链接起来，从而实现渐进式 的数据蒸馏(data distillation)。深度学习模型就像是数据处理的筛子，包含一系列越来越精细的 数据过滤器(即层)。
# 定义网络
本例中的网络包含 2 个 Dense 层，它们是密集连接(也叫全连接)的神经层。第二层(也 是最后一层)是一个 10 路 softmax 层，它将返回一个由 10 个概率值(总和为 1)组成的数组。 每个概率值表示当前数字图像属于 10 个数字类别中某一个的概率
```
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
```
要想训练网络，我们还需要选择编译(compile)步骤的三个参数。
- 损失函数(loss function):网络如何衡量在训练数据上的性能，即网络如何朝着正确的
方向前进。
- 优化器(optimizer):基于训练数据和损失函数来更新网络的机制。
- 在训练和测试过程中需要监控的指标(metric):本例只关心精度，即正确分类的图像所占的比例。
```
#编译阶段 使用rmsprop优化器，交叉熵作为损失函数，监控指标为准确度accuracy
firstnet.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
```
# 数据处理 
我们的数据格式与编写的网络不同，网络要求为n个28*28维的向量（2D张量），我们的是3D张量（60000，28，28）需要修改成（60000，28*28）
#并且将[0,255]的灰度范围映射到[0,1],并把数据类型由uint更改为float32
在开始训练之前，我们将对数据进行预处理，将其变换为网络要求的形状，并缩放到所 有值都在 [0, 1] 区间。比如，之前训练图像保存在一个 uint8 类型的数组中，其形状为 (60000, 28, 28)，取值区间为[0, 255]。我们需要将其变换为一个float32数组，其形 状为 (60000, 28 * 28)，取值范围为 0~1。
```
train_images=train_images.reshape(60000,28*28)
train_images=train_images.astype('float32')/255
test_images=test_images.reshape(10000,28*28)
test_images=test_images.astype('float32')/255
#我们还需要对标签进行分类编码，未来将会对这一步骤进行解释。
from keras.utils import to_categorical
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)
```
# 训练网络
现在我们准备开始训练网络，在 Keras 中这一步是通过调用网络的 fit 方法来完成的—— 我们在训练数据上拟合(fit)模型。
我们规定Batch_size=128 即每次迭代训练读取128个样本
5个epochs，指把整个数据集训练五遍
```
firstnet.fit(train_images,train_labels,epochs=5,batch_size=128)
```
可以看到由于数据集、网络规模较小，设备性能较强，训练很快
![image.png](https://upload-images.jianshu.io/upload_images/4064394-d20b3dbf3a5a1e15.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
训练过程中显示了两个数字:一个是网络在训练数据上的损失(loss)，另一个是网络在 训练数据上的精度(acc)。

我们很快就在训练数据上达到了 0.989(98.9%)的精度。现在我们来检查一下模型在测试 集上的性能。
```
test_loss,test_acc=firstnet.evaluate(test_images,test_labels)
print('test_loss: ',test_loss,'test_acc: ',test_acc)
```
![image.png](https://upload-images.jianshu.io/upload_images/4064394-ea490ed05c4dff17.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
- 测试集精度为 97.8%，比训练集精度低不少。训练精度和测试精度之间的这种差距是过拟 合(overfit)造成的。过拟合是指机器学习模型在新数据上的性能往往比在训练数据上要差，它 是第 3 章的核心主题。
- 第一个例子到这里就结束了。你刚刚看到了如何构建和训练一个神经网络，用不到 20 行的 Python 代码对手写数字进行分类
- 接下来详细介绍这个例子中的每一个步骤，并讲解其背后 的原理。接下来你将要学到张量(输入网络的数据存储对象)、张量运算(层的组成要素)和梯 6 度下降(可以让网络从训练样本中进行学习)。



# 参考文献
>[1] Python深度学习，（美）弗朗索瓦·肖莱，人民邮电出版社，2018，8



>本文内容：
1.神经网络数据表示
2.Numpy线性代数库
3.现实世界中的数据张量（图片、时序数据、视频等文件的张量表示方式）
4.张量运算 （神经网络中线性代数的编程实现）
代码仓库[https://github.com/MachinePlay/Keras-toturial](https://links.jianshu.com/go?to=https%3A%2F%2Fgithub.com%2FMachinePlay%2FKeras-toturial) 包含.py 代码和Jupyter Notebook的.ipynb文件

# 1. 数据表示
前面例子使用的数据存储在多维 Numpy 数组中，也叫张量(tensor)。一般来说，当前所有机器学习系统都使用张量作为基本数据结构。张量对这个领域非常重要，重要到 Google 的 TensorFlow 都以它来命名。那么什么是张量?

张量这一概念的核心在于，它是一个数据容器。它包含的数据几乎总是数值数据，因此它 是数字的容器。你可能对矩阵很熟悉，它是二维张量。张量是矩阵向任意维度的推广[注意， 张量的维度(dimension)通常叫作轴(axis)]。
## 1.1 标量（0D张量）
仅包含一个数字的张量叫作标量(scalar，也叫标量张量、零维张量、0D 张量)。在 Numpy 中，一个 float32 或 float64 的数字就是一个标量张量(或标量数组)。

你可以用 ndim 属性来查看一个 Numpy 张量的轴的个数。标量张量有 0 个轴(ndim == 0)。张量轴的个数也叫作 阶(rank)。下面是一个 Numpy 标量。
```
>>> import numpy as np 
>>> x = np.array(12) 
>>> x
array(12)
>>> x.ndim 0
```
## 1.2 向量（1D张量）
数字组成的数组叫作向量(vector)或一维张量(1D 张量)。一维张量只有一个轴。下面是 一个 Numpy 向量。
```
>>> x = np.array([12, 3, 6, 14, 7])
>>> x
array([12, 3, 6, 14, 7])
>>> x.ndim
1
```
这个向量有 5 个元素，所以被称为 5D 向量。不要把 5D 向量和 5D 张量弄混! 5D 向量只 有一个轴，沿着轴有 5 个维度，而 5D 张量有 5 个轴(沿着每个轴可能有任意个维度)。维度 (dimensionality)可以表示沿着某个轴上的元素个数(比如 5D 向量)，也可以表示张量中轴的个 数(比如 5D 张量)，这有时会令人感到混乱。对于后一种情况，技术上更准确的说法是 5 阶张量(张量的阶数即轴的个数)，但 5D 张量这种模糊的写法更常见。
## 1.3 矩阵(2D 张量)
向量组成的数组叫作矩阵(matrix)或二维张量(2D 张量)。矩阵有 2 个轴(通常叫作行和 列)。你可以将矩阵直观地理解为数字组成的矩形网格。下面是一个 Numpy 矩阵。
```
>>> x = np.array([[5, 78, 2, 34, 0],
                  [6, 79, 3, 35, 1],
                  [7, 80, 4, 36, 2]])
>>> x.ndim 2
```
第一个轴上的元素叫作行(row)，第二个轴上的元素叫作列(column)。在上面的例子中， [5, 78, 2, 34, 0] 是 x 的第一行，[5, 6, 7] 是第一列。
## 1.4 3D 张量与更高维张量
将多个矩阵组合成一个新的数组，可以得到一个 3D 张量，你可以将其直观地理解为数字组成的立方体。下面是一个 Numpy 的 3D 张量。
```
>>> x = np.array(
[[[5, 78, 2, 34, 0], 
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]], 
[[5, 78, 2, 34, 0], 
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]], 
[[5, 78, 2, 34, 0], 
[6, 79, 3, 35, 1],
[7, 80, 4, 36, 2]]])
>>> x.ndim 
 3
```
将多个 3D 张量组合成一个数组，可以创建一个 4D 张量，以此类推。深度学习处理的一般 是 0D 到 4D 的张量，但处理视频数据时可能会遇到 5D 张量。
## 1.5 关键属性
张量是由以下三个关键属性来定义的。
- 轴的个数(阶)。例如，3D 张量有 3 个轴，矩阵有 2 个轴。这在 Numpy 等 Python 库中
也叫张量的 ndim。
- 形状。这是一个整数元组，表示张量沿每个轴的维度大小(元素个数)。例如，前面矩阵示例的形状为 (3, 5)，3D 张量示例的形状为 (3, 3, 5)。向量的形状只包含一个
元素，比如 (5,)，而标量的形状为空，即 ()。
- 数据类型(在 Python 库中通常叫作 dtype)。这是张量中所包含数据的类型，例如，张量的类型可以是 float32、uint8、float64 等。在极少数情况下，你可能会遇到字符 (char)张量。注意，Numpy(以及大多数其他库)中不存在字符串张量，因为张量存储在预先分配的连续内存段中，而字符串的长度是可变的，无法用这种方式存储。

为了具体说明，我们回头看一下 MNIST 例子中处理的数据。首先加载 MNIST 数据集。 
```
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
接下来，我们给出张量 train_images 的轴的个数，即 ndim 属性。
>>> print(train_images.ndim)
3
```
下面是它的形状。
```
>>> print(train_images.shape) 
(60000, 28, 28)
```
下面是它的数据类型，即 dtype 属性。
```
>>> print(train_images.dtype)
uint8
```
所以，这里 train_images 是一个由 8 位整数组成的 3D 张量。更确切地说，它是 60 000个矩阵组成的数组，每个矩阵由 28×28 个整数组成。每个这样的矩阵都是一张灰度图像，元素 取值范围为 0~255。
我们用 Matplotlib 库(Python 标准科学套件的一部分)来显示这个 3D 张量中的第 4 个数字
```
from keras.datasets import mnist
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print(train_images.shape)
print(train_images.ndim)
print(train_images.dtype)
digit=train_images[4]
import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()
```
![image.png](https://upload-images.jianshu.io/upload_images/4064394-0f5b4616f3c4840d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 2.Nmupy张量操作
## 2.1 切片
在前面的例子中，我们使用语法 train_images[i] 来选择沿着第一个轴的特定数字。选 择张量的特定元素叫作张量切片(tensor slicing)。我们来看一下 Numpy 数组上的张量切片运算。
下面这个例子选择第 10~100 个数字(不包括第 100 个)，并将其放在形状为 (90, 28, 28) 的数组中。
```
>>> my_slice = train_images[10:100] 
>>> print(my_slice.shape)
(90, 28, 28)
```
它等同于下面这个更复杂的写法，给出了切片沿着每个张量轴的起始索引和结束索引。 注意，: 等同于选择整个轴。
```
>>> my_slice = train_images[10:100, :, :]
>>> my_slice.shape
(90, 28, 28)
>>> my_slice = train_images[10:100, 0:28, 0:28] 
>>> my_slice.shape
(90, 28, 28)
```
一般来说，你可以沿着每个张量轴在任意两个索引之间进行选择。例如，你可以在所有图 像的右下角选出 14 像素×14 像素的区域:
```
my_slice=train_images[:,14:,14:]
```
也可以使用负数索引。与 Python 列表中的负数索引类似，它表示与当前轴终点的相对位置。 你可以在图像中心裁剪出 14 像素×14 像素的区域:
```
my_slice=train_images[:,7:-7,7:-7]
```
![image.png](https://upload-images.jianshu.io/upload_images/4064394-a86ea4b29f39193d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
## 2.2 数据批量的概念
通常来说，深度学习中所有数据张量的第一个轴(0 轴，因为索引从 0 开始)都是样本轴(samples axis，有时也叫样本维度)。在 MNIST 的例子中，样本就是数字图像。
此外，深度学习模型不会同时处理整个数据集，而是将数据拆分成小批量。具体来看，下 面是 MNIST 数据集的一个批量，批量大小为 128。
```
batch = train_images[:128]
```
然后是下一个批量
```
batch= train_images[128:256]
```
然后是第 n 个批量
```
batch = train_images[128 * n:128 * (n + 1)]
```
对于这种批量张量，第一个轴(0 轴)叫作批量轴(batch axis)或批量维度(batch dimension)。 在使用 Keras 和其他深度学习库时，你会经常遇到这个术语。
批量（batch_size)的产生是妥协和计算的需要
# 3.现实世界中的数据张量
我们用几个你未来会遇到的示例来具体介绍数据张量。你需要处理的数据几乎总是以下类 别之一。
- 向量数据:2D 张量，形状为 (samples, features)。
- 时间序列数据或序列数据:3D 张量，形状为 (samples, timesteps, features)。  
- 图像:4D 张量，形状为 (samples, height, width, channels) 或 (samples, channels,height, width)。
- 视频:5D 张量，形状为 (samples, frames, height, width, channels) 或 (samples,frames, channels, height, width)。
## 3.1 向量数据
这是最常见的数据。对于这种数据集，每个数据点都被编码为一个向量，因此一个数据批 量就被编码为 2D 张量(即向量组成的数组)，其中第一个轴是样本轴，第二个轴是特征轴。
我们来看两个例子。
- 人口统计数据集，其中包括每个人的年龄、邮编和收入。每个人可以表示为包含 3 个值的向量，而整个数据集包含 100 000 个人，因此可以存储在形状为 (100000, 3) 的 2D张量中。
- 文本文档数据集，我们将每个文档表示为每个单词在其中出现的次数(字典中包含20 000 个常见单词)。每个文档可以被编码为包含 20 000 个值的向量(每个值对应于 字典中每个单词的出现次数)，整个数据集包含 500 个文档，因此可以存储在形状为 (500, 20000) 的张量中。
## 3.2 时间序列数据或序列数据
当时间(或序列顺序)对于数据很重要时，应该将数据存储在带有时间轴的 3D 张量中。 每个样本可以被编码为一个向量序列(即 2D 张量)，因此一个数据批量就被编码为一个 3D 张量
![image.png](https://upload-images.jianshu.io/upload_images/4064394-e9a15ada08ddfbd1.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

根据惯例，时间轴始终是第 2 个轴(索引为 1 的轴)。我们来看几个例子。
- 股票价格数据集。每一分钟，我们将股票的当前价格、前一分钟的最高价格和前一分钟 的最低价格保存下来。因此每分钟被编码为一个 3D 向量，整个交易日被编码为一个形 状为 (390, 3) 的 2D 张量(一个交易日有 390 分钟)，而 250 天的数据则可以保存在一个形状为 (250, 390, 3) 的 3D 张量中。这里每个样本是一天的股票数据。
- 推文数据集。我们将每条推文编码为 280 个字符组成的序列，而每个字符又来自于 128 个字符组成的字母表。在这种情况下，每个字符可以被编码为大小为 128 的二进制向量 (只有在该字符对应的索引位置取值为 1，其他元素都为 0)。那么每条推文可以被编码 为一个形状为 (280, 128) 的 2D 张量，而包含 100 万条推文的数据集则可以存储在一个形状为 (1000000, 280, 128) 的张量中。
## 3.3 图像数据
图像通常具有三个维度:高度、宽度和颜色深度。虽然灰度图像(比如 MNIST 数字图像) 只有一个颜色通道，因此可以保存在 2D 张量中，但按照惯例，图像张量始终都是 3D 张量，灰 度图像的彩色通道只有一维。因此，如果图像大小为 256×256，那么 128 张灰度图像组成的批 量可以保存在一个形状为 (128, 256, 256, 1) 的张量中，而 128 张彩色图像组成的批量则可以保存在一个形状为 (128, 256, 256, 3) 的张量中
![image.png](https://upload-images.jianshu.io/upload_images/4064394-144492c794c3c2ab.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

图像张量的形状有两种约定:`通道在后(channels-last)`的约定(在 TensorFlow 中使用)和 `通道在前(channels-first)`的约定(在 Theano 中使用)。Google 的 TensorFlow 机器学习框架将 颜色深度轴放在最后:(samples, height, width, color_depth)。与此相反，Theano 将图像深度轴放在批量轴之后:(samples, color_depth, height, width)。如果采用 Theano 约定，前面的两个例子将变成 (128, 1, 256, 256) 和 (128, 3, 256, 256)。 Keras 框架同时支持这两种格式。
## 3.4 视频数据
视频数据是现实生活中需要用到 5D 张量的少数数据类型之一。视频可以看作一系列帧， 6 每一帧都是一张彩色图像。由于每一帧都可以保存在一个形状为(height, width, color_ depth) 的 3D 张量中，因此一系列帧可以保存在一个形状为 (frames, height, width, color_depth) 的 4D 张量中，而不同视频组成的批量则可以保存在一个 5D 张量中，其形状为 (samples, frames, height, width, color_depth)。

举个例子，一个以每秒 4 帧采样的 60 秒 YouTube 视频片段，视频尺寸为 144×256，这个 视频共有 240 帧。4 个这样的视频片段组成的批量将保存在形状为 (4, 240, 144, 256, 3) 的张量中。总共有 106 168 320 个值!如果张量的数据类型(dtype)是 float32，每个值都是 32 位，那么这个张量共有 405MB。好大!你在现实生活中遇到的视频要小得多，因为它们不以 float32 格式存储，而且通常被大大压缩，比如 MPEG 格式。
# 4.张量运算
所有计算机程序最终都可以简化为二进制输入上的一些二进制运算(AND、OR、NOR 等)，与此类似，深度神经网络学到的所有变换也都可以简化为数值数据张量上的一些张量运算(tensoroperation)，例如加上张量、乘以张量等。
在最开始的例子中，我们通过叠加 Dense （全联接）层来构建网络。使用Keras库实现一个全联接层的实例如下所示。
```
keras.layers.Dense(512, activation='relu')
```
这个层可以理解为一个函数，输入一个 2D 张量，返回另一个 2D 张量，即输入张量的新 表示。具体而言，这个函数如下所示(其中 W 是一个 2D 张量，b 是一个向量，二者都是该层的 属性)。
 ${f(x)=relu(W*X+b)}$
```
output = relu(dot(W, input) + b)
```
relu函数图像如图所示
![image.png](https://upload-images.jianshu.io/upload_images/4064394-4e1fa201ff43fc5e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
我们将上式拆开来看。这里有三个张量运算:输入张量和张量 W 之间的点积运算(dot)、
得到的 2D 张量与向量 b 之间的加法运算(+)、最后的 relu 运算。relu(x) 是 max(x, 0)。
这里实际上就是实现了一层并行排列神经元，是神经网络的一层，将一个2D张量输入一层神经元，经过神经元的激活函数运算后将结果输出给下一层
![image.png](https://upload-images.jianshu.io/upload_images/4064394-c5bdee6b427ce333.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/4064394-90eb1c1965b75387.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

如果以mnist数据为例，其中向量X（x1,x2,···,xn)就是张图片的28$*$28个维度的元素一字排开（相当于每个像素灰度值依次排开），共有60000张，组成了形状为（60000，28${*}$28）的张量，张量W就是这每接收一张图片这一层层神经元的权重，向量b就是这一层神经元的阈值，每个神经元的输入数据乘以权重后与阈值做差的结果，再经过激活函数（选择合适的激活函数也是个很大的问题）的映射，就是这一个神经元的输出y，经过很多神经元组成的网络结构后，输出一个最终预测结果y～，我们通过一些数学方法衡量预测结果y～与真实结果y的差距（称为损失函数），根据这个差距调整所有神经元里面的权重（称为优化算法、优化器，比如非常有名的BP算法），这个过程称为训练，经过大量数据训练后，网络中的权重能够保证输出质量较高的预测结果y～，损失函数的值也因此降低。这个过程其实是一个简单的反馈过程。
这个过程将在下一章讲述
## 4.1 逐元素运算
我们使用relu函数作为神经元的激活函数，也被称为线性修正单元relu 运算和加法都是逐元素(element-wise)的运算，即该运算独立地应用于张量中的每 个元素，也就是说，这些运算非常适合大规模并行实现(向量化实现，这一术语来自于 1970— 1990 年间向量处理器超级计算机架构)。我们可以借助线性代数完成运算。

如果你想对逐元素运算编写简单的 Python 实现，那么 可以用 for 循环。下列代码是对逐元素 relu 运算的简单实现。
```
def naive_relu(x):
    assert len(x.shape)==2 ##输入的x是一个2D张量（sample，feature）
    x=x.copy()#避免张量被覆盖
    for i in (x.shape[0]):# 对每一个样本
        for j in (x.shape[1]):#的每一个元素
            x[i,j]=max(x[i,j],0)
    return x
```
对于加法采用同样的实现方法。
```
def naive_add(x,y):
    assert len(x.shape)==2
    assert len(y.shape)==2
    x=x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j]+=y[i,j]
    return x
```
根据同样的方法，你可以实现逐元素的乘法、减法等。

在实践中处理 Numpy 数组时，我们不必自己编写这些基本线性代数运算（包括后面的，下面只是讲述如果自己写应该如何实现），这些运算都是优化好的 Numpy 内置函数，这些函数将大量运算交给安装好的基础线性代数子程序(BLAS，basic linear algebra subprograms)实现(没装 的话，应该装一个)。BLAS 是低层次的、高度并行的、高效的张量操作程序，通常用 Fortran 或 C 语言来实现。

因此，在 Numpy 中可以直接进行下列逐元素运算，速度非常快。
```
import numpy as np
 z=x+y #逐元素的相加 
 z = np.maximum(z, 0.)#relu(z)
```

## 4.2广播
上一节 naive_add 的简单实现仅支持两个形状相同的 2D 张量相加。但在前面介绍的 Dense 层中，我们将一个 2D 张量与一个向量相加。如果将两个形状不同的张量相加，会发生 什么?

如果没有歧义的话，较小的张量会被广播(broadcast)，以匹配较大张量的形状。广播包含 4 以下两步。
- (1) 向较小的张量添加轴(叫作广播轴)，使其ndim与较大的张量相同。
- (2) 将较小的张量沿着新轴重复，使其形状与较大的张量相同。

来看一个具体的例子。假设 X 的形状是 (32, 10)，y 的形状是 (10,)。首先，我们给 y添加空的第一个轴，这样 y 的形状变为 (1, 10)。然后，我们将 y 沿着新轴重复 32 次，这样 得到的张量 Y 的形状为 (32, 10)，并且 Y[i, :] == y for i in range(0, 32)。现在， 我们可以将 X 和 Y 相加，因为它们的形状相同。

在实际的实现过程中并不会创建新的 2D 张量，因为那样做非常低效。重复的操作完全是 虚拟的，它只出现在算法中，而没有发生在内存中。但想象将向量沿着新轴重复 10 次，是一种 很有用的思维模型。下面是一种简单的实现。
```
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2 #x 是一个 Numpy 的 2D 张量
    assert len(y.shape) == 1 #y 是一个 Numpy 向量
    assert x.shape[1] == y.shape[0]
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]): x[i, j] += y[j]
    return x
```
如果一个张量的形状是 (a, b, ... n, n+1, ... m)，另一个张量的形状是 (n, n+1, ... m)，那么你通常可以利用广播对它们做两个张量之间的逐元素运算。广播操作会自动应用 于从a到n-1的轴。

下面这个例子利用广播将逐元素的 maximum 运算应用于两个形状不同的张量。
```
import numpy as np
x=np.random.random((64,3,32,10)) #x 是形状为 (64, 3, 32, 10) 的随机张量
y=np.random.random((32,10)) #y是形状为(32,10)的随机张量
z=np.maximum(x,y) #输出的形状与x相同(64, 3, 32, 10) 
```
## 4.3 张量点积运算（矩阵点乘）
点积运算，也叫张量积(tensor product，不要与逐元素的乘积弄混)，是最常见也最有用的 张量运算。与逐元素的运算不同，它将输入张量的元素合并在一起。在 Numpy、Keras、Theano 和 TensorFlow 中，都是用 * 实现逐元素乘积。TensorFlow 中的 点积使用了不同的语法，但在 Numpy 和 Keras 中，都是用标准的 dot 运算符来实现点积。
```
import numpy as np
z=np.dot(x,y)
```
数学符号中的点(.)表示点积运算。 z=x.y

从数学的角度来看，点积运算做了什么?我们首先看一下两个向量 x 和 y 的点积。其计算 过程如下。
```
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i] 
    return z
```

注意，两个向量之间的点积是一个标量，而且只有元素个数相同的向量之间才能做点积。

你还可以对一个矩阵 x 和一个向量 y 做点积，返回值是一个向量，其中每个元素是 y 和 x 的每一行之间的点积。其实现过程如下。
```
import numpy as np
def naive_matrix_vector_dot(x, y): 
    assert len(x.shape) == 2 #x是一个Numpy矩阵
    assert len(y.shape) == 1 #y是一个Numpy向量
    assert x.shape[1] == y.shape[0] #x的第1维必须和y的第0维一样
    z=np.zeros(x.shape[0])
    for i in range(x.shape[0]):#每一行的
        for j in range(x.shape[1]):#每一列   
            z[i]=x[i,j]*y[j]
            return z #返回类型与x.shape[0]相同
```
你还可以复用前面写过的代码，从中可以看出矩阵 - 向量点积与向量点积之间的关系。
```
def naive_matrix_vector_dot(x, y): 
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        z[i] = naive_vector_dot(x[i, :], y) 
    return z
```
注意，如果两个张量中有一个的 ndim 大于 1，那么 dot 运算就不再是对称的，也就是说，注意，如果两个张量中有一个的 ndim 大于 1，那么 dot 运算就不再是对称的，也就是说，dot(x, y) 不等于 dot(y, x)。
这很好理解，矩阵相乘不满足交换律也是这个道理。

当然，点积可以推广到具有任意个轴的张量。最常见的应用可能就是两个矩阵之间的点积。 对于两个矩阵 x 和 y，当且仅当 x.shape[1] == y.shape[0] 时，你才可以对它们做点积(dot(x, y))。得到的结果是一个形状为 (x.shape[0], y.shape[1]) 的矩阵，其元素为 x 的行与 y 的列之间的点积。其简单实现如下。
```
def naive_matrix_dot(x,y):
    assert len(x.shape[0])==2
    assert len(y.shape[1])==2
    assert x.shape[1]=y.shape[0] #x的列数等于y的行数
    z=np.zeros((x.shape[0],y.shape[1])) #提前构造好全零矩阵
    for i in range(x.shape[0]):
        for j in range(y.shape[1):
            row_x=x[i,:]
            col_y=y[:,j]
            z[i,j]=naive_vector_dot(row_x,col_y)
    return z
```
也就是一个标准的矩阵点乘过程。如果是更高维的张量做点积，其形状匹配遵循上述原则：
```
    （a,b,c,d).(d,)=（a,b,c)
     (a,b,c,d).(d,e)=(a,b,c,e)
```
## 4.4 张量变形
第三个重要的张量运算是张量变形(tensor reshaping)。虽然前面神经网络第一个例子的 Dense 层中没有用到它，但在将图像数据输入神经网络之前，我们在预处理时用到了这个运算。
```
train_images = train_images.reshape((60000, 28 * 28))
```
我们把（60000,28,28)的ndim=3张量变形为（60000,28*28)的ndim=2的张量
张量变形是指改变张量的行和列，以得到想要的形状。变形后的张量的元素总个数与初始 张量相同。简单的例子可以帮助我们理解张量变形。
```
>>> x = np.array([[0., 1.], [2., 3.],
[4., 5.]]) >>> print(x.shape)
(3, 2)
>>> x = x.reshape((6, 1)) >>> x
array([[ 0.],
           [ 1.],
           [ 2.],
           [ 3.],
           [ 4.],
           [ 5.]])
>>> x = x.reshape((2, 3)) >>> x
array([[ 0., 1., 2.],
[ 3., 4., 5.]])
```
经常遇到的一种特殊的张量变形是转置(transposition)。对矩阵做转置是指将行和列互换， 使 x[i, :] 变为 x[:, i]。
 ```
>>> x = np.zeros((300, 20)) 
>>> x = np.transpose(x) 
>>> print(x.shape)
(20, 300)
```
## 4.5 张量运算的几何解释
对于张量运算所操作的张量，其元素可以被解释为某种几何空间内点的坐标，因此所有的张量运算都有几何解释。
通常来说，仿射变换、旋转、缩放等基本的几何操作都可以表示为张量运算。举个例子，要将 一个二维向量旋转 theta 角，可以通过与一个 2×2 矩阵做点积来实现，这个矩阵为 R = [u, v]，其 中 u 和 v 都是平面向量:u = [cos(theta), sin(theta)]，v = [-sin(theta), cos(theta)]。
## 4.6深度学习的几何解释
前面讲过，神经网络完全由一系列张量运算组成，而这些张量运算都只是输入数据的几何 变换。因此，你可以将神经网络解释为高维空间中非常复杂的几何变换，这种变换可以通过许 多简单的步骤来实现。

对于三维的情况，下面这个思维图像是很有用的。想象有两张彩纸:一张红色，一张蓝色。
![解开复杂的数据流形](https://upload-images.jianshu.io/upload_images/4064394-527c5b3eabaceae7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

将其中一张纸放在另一张上。现在将两张纸一起揉成小球。这个皱巴巴的纸球就是你的输入数 据，每张纸对应于分类问题中的一个类别。神经网络(或者任何机器学习模型)要做的就是找 到可以让纸球恢复平整的变换，从而能够再次让两个类别明确可分。通过深度学习，这一过程 可以用三维空间中一系列简单的变换来实现，比如你用手指对纸球做的变换，每次做一个动作。

让纸球恢复平整就是机器学习的内容:为复杂的、高度折叠的数据流形找到简洁的表示。 现在你应该能够很好地理解，为什么深度学习特别擅长这一点:它将复杂的几何变换逐步分解 为一长串基本的几何变换，这与人类展开纸球所采取的策略大致相同。深度网络的每一层都通 过变换使数据解开一点点——许多层堆叠在一起，可以实现非常复杂的解开过程。

下一章将重温神经网络传播算法，主要复习微分、基于梯度的优化、反向传播等原理
下周人工智能考试可能不会更新

# 5.参考文献：
> [1]《机器学习》周志华著，清华大学出版社,2016. 
[2]  Python深度学习，（美）弗朗索瓦·肖莱，人民邮电出版社，2018，8





