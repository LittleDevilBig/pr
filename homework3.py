import numpy as np
import matplotlib.pyplot as plt

w1 = np.array([[0.1, 1.1], [6.8, 7.1], [-3.5, -4.1], [2.0, 2.7], [4.1, 2.8],
              [3.1, 5.0], [-0.8, -1.3], [0.9, 1.2], [5.0, 6.4], [3.9, 4.0]], dtype=np.float32)
w2 = np.array([[7.1, 4.2], [-1.4, -4.3], [4.5, 0.0], [6.3, 1.6], [4.2, 1.9],
               [1.4, -3.2], [2.4, -4.0], [2.5, -6.1], [8.4, 3.7], [4.1, -2.2]], dtype=np.float32)
w3 = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2],
               [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1]], dtype=np.float32)
w4 = np.array([[-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0],
               [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]], dtype=np.float32)


def augment(data):
    m, _ = data.shape
    data = np.insert(data, 0, np.ones([1, m]), axis=1)
    return data


def batch_perception(w1, w2):
    x1 = augment(w1)
    x2 = augment(w2)
    x3 = np.negative(x2)
    data = np.r_[x1, x3]
    a = np.zeros(3)
    n = 1
    k = 0
    theta = 0.00001
    while True:
        k += 1
        j = np.dot(a, data.T)
        y = data[j <= 0, :]
        if np.abs(n*np.sum(y)) < theta:
            break
        a += n*np.sum(y, axis=0)
    print(k)
    plt.scatter([x for [x, _] in w1], [y for [_, y] in w1])
    plt.scatter([x for [x, _] in w2], [y for [_, y] in w2])
    x = np.linspace(-5, 10, 256)
    y = (-a[0] - a[1] * x) / a[2]
    plt.plot(x, y)
    plt.show()


def ho_kashyap(w1, w2):
    x1 = augment(w1)
    x2 = augment(w2)
    x3 = np.negative(x2)
    data = np.r_[x1, x3]
    a = np.ones(3)
    b = np.ones(20)
    y_t = np.linalg.pinv(data)
    n = 0.05
    k = 0
    theta = 0.001
    while True:
        k += 1
        e = np.dot(data, a) - b
        j = e
        e = e+np.abs(e)
        if np.max(e) < theta:
            break
        b += n*e
        a = np.dot(y_t, b)
    print(k)
    print(np.linalg.norm(j))
    plt.scatter([x for [x, _] in w1], [y for [_, y] in w1])
    plt.scatter([x for [x, _] in w2], [y for [_, y] in w2])
    x = np.linspace(-8, 8, 256)
    y = (-a[0] - a[1] * x) / a[2]
    plt.plot(x, y)
    plt.show()


def mse(train,test):
    train_y=np.zeros([4,32])
    test_y=np.zeros([4,8])
    for i in range(4):
        train_y[i, 8*i:8*(i+1)] = 1
        test_y[i, 2*i:2*(i+1)] = 1
    w=np.linalg.pinv(train).T.dot(train_y.T)
    y=w.T.dot(test)
    print(np.argmax(y,axis=0))


if __name__ == '__main__':
    # batch_perception(w1,w2)
    # ho_kashyap(w2,w4)
    x1 = augment(w1)
    x2 = augment(w2)
    x3 = augment(w3)
    x4 = augment(w4)
    print(x4.shape)
    train=np.c_[x1[:8].T,x2[:8].T,x3[:8].T,x4[:8].T]
    test=np.c_[x1[8:].T,x2[8:].T,x3[8:].T,x4[8:].T]
    mse(train,test)
