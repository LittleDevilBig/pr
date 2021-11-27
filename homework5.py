import numpy as np
from scipy.io import loadmat

class Classiier(object):
    def __init__(self,configs) :
        self.file=configs['file']
        self.method=configs['method']
        self.eig_num=configs['eig_num']
        self.knum=configs['knum']
        self.train_data,self.test_data,self.train_label,self.test_label=self.load_data()
        self.normalization()

    def load_data(self):
        if self.file=='./ORLData_25.mat':
            file=loadmat(self.file)
            data=file['ORLData'].T.astype(np.float32)[:,:-1]
            label=file['ORLData'].T.astype(int)[:,-1]
            idx= np.arange(400)
            train_data,test_data=data[idx%10<8,:],data[idx%10>7,:]
            train_label,test_label=label[idx%10<8],label[idx%10>7]
            return train_data,test_data,train_label,test_label
        else: 
            self.file=='./vehicle.mat'
            file=loadmat(self.file)
            data=file['UCI_entropy_data']['train_data'][0,0].T.astype(np.float32)[:,:-1]
            label=file['UCI_entropy_data']['train_data'][0,0].T.astype(int)[:,-1]
            train_data,test_data=data[:676,:],data[676:,:]
            train_label,test_label=label[:676],label[676:]
            return train_data,test_data,train_label,test_label
    
    def normalization(self):
        mu=np.mean(self.train_data,axis=0)
        sigma=np.std(self.train_data,axis=0)
        self.train_data=(self.train_data-mu)/sigma
        self.test_data=(self.test_data-mu)/sigma

    def get_cov(self,data_):
        data=data_.copy()
        data-=np.mean(data,axis=0)
        num=data.shape[0]
        cov=np.dot(data.T,data)/(num-1)
        return cov

    def get_eig(self,cov):
        eig_val,eig_vec=np.linalg.eig(cov)
        w=eig_vec[:,:self.eig_num]
        return np.real(w)

    def pca(self):
        cov=self.get_cov(self.train_data)
        w=self.get_eig(cov)
        self.train_data=np.dot(self.train_data,w)
        self.test_data=np.dot(self.test_data,w)

    def lda(self):
        mu_s,cov_s=[],[]
        labels=np.unique(self.train_label)
        for i in range(labels.shape[0]):
            data=self.train_data[self.train_label==labels[i]]
            mu_s.append(np.mean(data,axis=0))
            cov_s.append(self.get_cov(data))
        mu_s=np.array(mu_s)
        cov_s=np.array(cov_s)
        Sb,Sw=np.zeros_like(cov_s[0]),np.zeros_like(cov_s[0])
        for i in range(labels.shape[0]):
            sb=np.dot(np.expand_dims(mu_s[i],0).T,np.expand_dims(mu_s[i],0))
            sw=cov_s[i]
            Sb+=sb
            Sw+=sw
        data_cov=np.dot(np.linalg.inv(Sw),Sb)
        w=self.get_eig(data_cov)
        self.train_data=np.dot(self.train_data,w)
        self.test_data=np.dot(self.test_data,w)

    def knn(self):
        train_num,test_num=self.train_data.shape[0],self.test_data.shape[0]
        d=np.zeros([test_num,train_num])
        for i in range(test_num):
            for j in range(train_num):
                d[i][j]=np.linalg.norm(self.test_data[i]-self.train_data[j])
        sort=np.argsort(d,axis=1)
        kn=sort[:,:self.knum]
        count=0
        for i in range(test_num):
            label_k=np.argmax(np.bincount(self.train_label[kn[i]]))
            label_true=self.test_label[i]
            if label_k==label_true:
                count+=1
        acc=count/test_num
        print (acc)

    def run(self):
        if self.method=='pca':
            self.pca()
        else:
            self.lda()
        self.knn()

if __name__=='__main__':
    configs={
        'file':'./ORLData_25.mat',
        #'file':'./vehicle.mat',
        'eig_num':40,
        'method':'lda',
        'knum':1
    }
    classifier=Classiier(configs)
    classifier.run()