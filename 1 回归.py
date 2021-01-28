import os
os.environ["KERAS_BACKEND"] = "tensorflow"
##import keras
import numpy as np
np.random.seed(1337)
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 创建一些数据
X = np.linspace(-1,1,200)
np.random.shuffle(X) # 打乱数据
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))

# 展示数据
##plt.scatter(X, Y)
##plt.show()

X_train, Y_train = X[:160], Y[:160] # 前160个数据作为训练数据
X_test, Y_test = X[160:], Y[160:]   # 后40个数据作为测试数据

# 创建一个神经网络（第一层到最后一层）
model = Sequential()
model.add(Dense(units=1,input_dim=1))

# 选择loss函数和优化器
model.compile(loss="mse", optimizer="sgd")
# 训练
print("training...")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print("train cost:", cost)

# 测试
cost = model.evaluate(X_test, Y_test, batch_size=40)
print("test cost:", cost)
W, b = model.layers[0].get_weights()
print("Weights=", W, "biases=", b)
