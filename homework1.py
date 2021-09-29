import numpy as np

pca_dim = 10


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def getData():
    train_data_dict = unpickle('./cifar-10-batches-py/data_batch_1')
    train_label = np.array(train_data_dict[b'labels'])
    train_data = np.array(train_data_dict[b'data'])
    train_data = train_data.astype(np.float32) / 255.0
    test_data_dict = unpickle('./cifar-10-batches-py/test_batch')
    test_label = np.array(test_data_dict[b'labels'])
    test_data = np.array(test_data_dict[b'data'])
    test_data = test_data.astype(np.float32) / 255.0
    return train_data, train_label, test_data, test_label


def getCov(data):
    n = data.shape[0]
    cov = np.dot(data.T, data) / (n - 1)
    return cov


def getEig(data):
    data = data - np.mean(data, axis=0)
    data_cov = getCov(data)
    _, eigvec = np.linalg.eig(data_cov)
    eig = eigvec[:, :pca_dim]
    return eig


def pca(data, mu, eig):
    data = data - mu
    data_pca = np.dot(data, eig)
    return data_pca


def cal(data, cov):
    data = data[None]
    ans = np.einsum('ijn, jk->ikn', data.transpose(0, 2, 1),
                    np.linalg.inv(cov))
    ans = np.einsum('ikn, jnk->ijn', ans, data)
    ans = np.squeeze(ans)
    return -ans


def LDF(data, mu_x, mu_y, p_x, p_y, eig):
    data_x = pca(data, mu_x, eig)
    data_y = pca(data, mu_y, eig)
    g_0 = cal(data_x, cov) + p_x
    g_1 = cal(data_y, cov) + p_y
    rate = np.sum(g_0 > g_1) / g_0.shape[0]
    return rate


def QDF(data, mu_x, mu_y, det_x, det_y, cov_x, cov_y, eig):
    data_x = pca(data, mu_x, eig)
    data_y = pca(data, mu_y, eig)
    g_0 = cal(data_x, cov_x) - det_x
    g_1 = cal(data_y, cov_y) - det_y
    rate = np.sum(g_0 > g_1) / g_0.shape[0]
    return rate


if __name__ == '__main__':
    class_x = 0
    class_y = 9
    train_data, train_label, test_data, test_label = getData()
    # 保存两类的训练集测试集
    train_data_x = train_data[train_label == class_x]
    train_data_y = train_data[train_label == class_y]
    train_data = np.append(train_data_x, train_data_y, axis=0)
    test_data_x = test_data[test_label == class_x]
    test_data_y = test_data[test_label == class_y]
    # 计算先验概率
    p_x = np.log(train_data_x.shape[0] / train_data.shape[0])
    p_y = np.log(train_data_y.shape[0] / train_data.shape[0])
    # 计算均值
    mu = np.mean(train_data, axis=0)
    mu_x = np.mean(train_data_x, axis=0)
    mu_y = np.mean(train_data_y, axis=0)
    # 计算特征向量
    eig = getEig(train_data)
    # 计算协方差
    cov = getCov(pca(train_data, mu, eig))
    cov_x = getCov(pca(train_data_x, mu_x, eig))
    cov_y = getCov(pca(train_data_y, mu_y, eig))
    # np.linalg.slogdet计算行列式的对数不溢出
    (_, det_x) = np.linalg.slogdet(cov_x)
    (_, det_y) = np.linalg.slogdet(cov_y)
    # 分类
    rate1 = LDF(test_data_x, mu_x, mu_y, p_x, p_y, eig)
    rate2 = 1-LDF(test_data_y, mu_x, mu_y, p_x, p_y, eig)
    rateldf = (rate1 + rate2) / 2
    print(rateldf)
    rate1 = QDF(test_data_x, mu_x, mu_y, det_x, det_y, cov_x, cov_y, eig)
    rate2 = 1-QDF(test_data_y, mu_x, mu_y, det_x, det_y, cov_x, cov_y, eig)
    rateqdf = (rate1 + rate2) / 2
    print(rateqdf)
