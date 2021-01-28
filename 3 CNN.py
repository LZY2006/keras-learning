import numpy as np
np.random.seed(1337) # 伪随机以保持与老师一致
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# 下载mnist数据集保存到"c:\users\用户名\.keras\datasets"(如果没有数据集)
# X的形状(60,000 28x28), Y的形状(10,000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 1, 28, 28)# / 255
x_test = x_test.reshape(-1, 1, 28, 28)# / 255

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 构建卷积神经网络
model = Sequential()

# 卷积层1 输出:(32, 28, 28)
model.add(Conv2D(
        filters=32,
        kernel_size = (5,5),
        padding="same",  # padding模式
        input_shape=(1,28,28) # 通道个数 长 宽
    ))
model.add(Activation("relu"))

# 池化层1 输出:(32, 14, 14)
model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=(2, 2),       # 步长
        padding="same"
    ))

# 卷积层2 输出:(64, 14, 14)
model.add(Conv2D(64, (5, 5), padding="same"))
model.add(Activation("relu"))

# 池化层2 输出:(64, 7, 7)
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

# 全连接层1 输入:(64 * 7 * 7) = 3136, 输出: 1024
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation("relu"))
# 全连接层 2 输入：1024 输出： 10(softmax)
model.add(Dense(10))
model.add(Activation("softmax"))
# 定义优化器
adam = Adam(lr=1e-4)

# 加入metrics来获得更多结果
model.compile(optimizer=adam, loss="categorical_crossentropy",
              metrics=["accuracy"])

# 训练模型
print("正在训练. . .")
model.fit(x_train, y_train, epochs=1, batch_size=32)

# 测试模型和先前定义的metrics
print("正在测试. . .")
loss, accuracy = model.evaluate(x_test, y_test)

print("test loss:", loss)
print("test accuracy:", accuracy)
