import numpy as np
np.random.seed(1337) # 伪随机以保持与老师一致
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop, Adam

# 下载mnist数据集保存到"c:\users\用户名\.keras\datasets"（如果没有数据集）
# X的形状(60,000 28x28), Y的形状(10,000,)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(x_train.shape[0], -1) / 255
x_test = x_test.reshape(x_test.shape[0], -1) / 255

y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 另一种方法来构建神经网络
model = Sequential([
        Dense(32, input_dim=784),
        Activation("relu"),
        Dense(10),
        Activation("softmax")
    ])
# 另一种方式来定义优化器
rmsprop = RMSprop()

# 加入metrics来获得更多结果
model.compile(
        optimizer=rmsprop,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
print("开始训练. . .")
model.fit(x_train, y_train, epochs=3, batch_size=32)
print("开始测试. . .")
loss, accuracy = model.evaluate(x_test, y_test)

print("test loss:", loss)
print("test accuracy:", accuracy)
