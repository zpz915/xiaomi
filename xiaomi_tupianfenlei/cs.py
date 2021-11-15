import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib   #关于路径的包 也方便管理
import  random
import IPython.display as display#作为作图用

#先跑一个baseline 进行一个多分类

data_dir = "/home/mi/zpz/opptpict/category_images"

data_root = pathlib.Path(data_dir)

print(data_root)


# for item in data_root.iterdir():
#      print(item)                    #查看当前路径下所有的文件

all_image_path= list(data_root.glob("*/*/*/"))  #用正则表达式提取所有的图片路径

# print(len(all_image_path))
# print(all_image_path[:3])
# print(all_image_path[-3:])

all_image_path = [str(path) for path in all_image_path] #把路径转化为真正的路径

# print(all_image_path[10:12])

#把照片进行乱序操作
random.shuffle(all_image_path)
# print(all_image_path[10:12])



image_count = len(all_image_path)
# print(image_count)

label_names = sorted(item.name for item in data_root.glob("*/")) #列出所有的以及分类的名字

# print(label_names)

label_to_index = dict((name,index) for index,name in enumerate(label_names)) #给类别编码
index_to_label = dict((v,k) for k,v in label_to_index.items())
#print(label_to_index)
#print(index_to_label)

all_image_label = [label_to_index[pathlib.Path(p).parent.parent.name] for p in all_image_path] #通过目录确定类型 然后根据字典进行编码

# print(all_image_label[:5])
# print(all_image_path[:5])
# for n in range(3):
#     image_index = random.choice(range(len(all_image_path))) #随机选一个数字
#     display.display(display.Image(filename=all_image_path[image_index]))#随即选择一个图进行展示
#     print(index_to_label[all_image_label[image_index]]) #显示label
#     print()
#
# for n in range(3):
#     image_index = random.choice(range(len(all_image_path))) #随机选一个数字
#     plt.imshow(all_image_path[image_index])#随即选择一个图进行展示
#     plt.show()
#     print(index_to_label[all_image_label[image_index]]) #显示label
#####图片处理
img_path = all_image_path[2] #路径
#print(img_path)
img_raw = tf.io.read_file(img_path)  #tf 读取图片的方法 二进制显示
#print(img_raw)
img_tensor = tf.image.decode_image(img_raw) #解码
# print(img_tensor.shape) #查看解码成功图片的形状  不同图片解码形状不一样
#
# print(img_tensor.dtype)
# print(img_tensor)

img_tensor = tf.cast(img_tensor,tf.float32) #转化数据类型 float32

img_tensor = img_tensor/255 #进行标准化

# print(img_tensor.numpy())
# print(img_tensor.numpy().max())

def load_preprocess_image(img_path):    #综合以上内容 写出图片预处理函数
    img_raw = tf.io.read_file(img_path)
    img_tensor = tf.image.decode_jpeg(img_raw,channels=3) #因为图片的格式jpeg  解码专业的 彩色图片对应通道书是3
    img_tensor = tf.image.resize(img_tensor,[256,256])  #改变图片大小方法
    img_tensor = tf.cast(img_tensor, tf.float32)
    img = img_tensor / 255
    return img


# img_path = all_image_path[50]
# plt.imshow(load_preprocess_image(img_path)) #这种显示图片的方法必须是经过预处理的 这时候图片的数据类型已经是float类型
# plt.show()

#建立tf.data类型  标签和输入
path_ds = tf.data.Dataset.from_tensor_slices(all_image_path)
image_dataset = path_ds.map(load_preprocess_image) #映射其他函数
label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)

# for label in label_dataset.take(10): #打印前10个类别
#     print(label.numpy())

# for image in image_dataset.take(1):
#     print(image)
# print(image_dataset.take(1))
#print(image_dataset)


dataset = tf.data.Dataset.zip((image_dataset,label_dataset))  #把对应标签和图片关联在一起

#print(dataset)

#在dataset中划分训练集和测试集

test_count = int(image_count*0.2)
train_count = image_count-test_count

# print(test_count)
# print(train_count)

train_dataset = dataset.skip(test_count)  #skip方法开始划分训练集和测试集
test_dataset = dataset.take(test_count)   #测试集


#一个batch一个batch来进行取
BATCH_SIZE = 32

train_dataset = train_dataset.repeat().shuffle(buffer_size = train_count).batch(BATCH_SIZE)   #打乱 重复 每一批次 repeat方法可以使dataset源源不断产生数据

test_dataset = test_dataset.batch(BATCH_SIZE)



#构造模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(256,256,3),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(64,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D()) #默认缩小一半
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(128,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D()) #默认缩小一半
model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.Conv2D(256,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(512,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.MaxPool2D())
model.add(tf.keras.layers.Conv2D(1024,(3,3),activation="relu",padding="same"))
model.add(tf.keras.layers.GlobalAveragePooling2D())#全居池化
model.add(tf.keras.layers.Dense(1024,activation="relu"))
model.add(tf.keras.layers.Dense(256,activation="relu"))
model.add(tf.keras.layers.Dense(40,activation="softmax")) #40个类别


model.compile(optimizer="adam",
             loss="sparse_categorical_crossentropy",
             metrics=["acc"])

steps_per_epoch = train_count//BATCH_SIZE
validation_steps = test_count//BATCH_SIZE

history = model.fit(train_dataset,epochs=10,steps_per_epoch=steps_per_epoch,validation_data = test_dataset,validation_steps=validation_steps)


history.history.keys()
plt.plot(history.epoch,history.history.get("acc"),label="acc")
plt.plot(history.epoch,history.history.get("val_acc"),label="val_acc")
plt.legend()  #画图   有下图看出是一个过拟合

plt.plot(history.epoch,history.history.get("loss"),label="loss")
plt.plot(history.epoch,history.history.get("val_loss"),label="val_loss")
plt.legend()  #画图