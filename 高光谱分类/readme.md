# 高光谱分类叙述文档

## Step1.数据归一化

```python
import scipy.io as scio#这个库用来读.mat文件
import numpy as np
import pandas as pd
data = scio.loadmat('data2_train.mat')
test = scio.loadmat('data_test_final.mat')
x=data['data2_train']
X=test['data_test_final']
y=np.zeros([x.shape[0],1],dtype=np.uint16)
y[y==0]=2#自己创建标签
Data=np.concatenate([x,y],axis=1)#拼接
for i in [3,5,6,8,10,11,12,14]:
    data = scio.loadmat('data'+str(i)+'_train.mat')
    x=data['data'+str(i)+'_train']
    y=np.zeros([x.shape[0],1],dtype=np.uint16)
    y[y==0]=i
    data=np.concatenate([x,y],axis=1)
    Data=np.concatenate([Data,data],axis=0)

D=pd.DataFrame(Data)
X=pd.DataFrame(X)
data=D.iloc[:,200]#这是标签，存储为labels.csv文件
D=D.drop(columns=200)

df_norm = (D - D.min()) / (D.max() - D.min())#这是归一化后的训练集x，存储为x.csv文件
tf_norm = (X - X.min()) / (X.max() - X.min())#这是归一化后的测试集x，存储为test-x.csv文件
```

## step2.PCA降维

```python
import pandas as pd
from sklearn.decomposition import PCA

data=pd.read_csv('x.csv',index_col=0)
test=pd.read_csv('test-x.csv',index_col=0)

pca = PCA(n_components=40)#注1
D=pca.fit_transform(data)#这是降维后的训练集x，存储为pca-x.csv文件
T=pca.fit_transform(test)#这是降维后的测试集x，存储为pca-test-x.csv文件
```

注1：从200维降成40维是因为，降成40维后包含了源数据99%的信息

10维：96.72%	20维：98.11% 	30维：98.87%	40维：99.28%



![](https://github.com/m-L-0/18b-lidongliang--2016-449/blob/master/%E9%AB%98%E5%85%89%E8%B0%B1%E5%88%86%E7%B1%BB/image/PCA.png)

## step3.训练模型

```python
from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split


x=pd.read_csv('pca-x.csv',index_col=0)
y=pd.read_csv('labels.csv',index_col=0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = SVC(gamma='auto',C=120)#手动调参10,20,30,40,50，.....120,130,140
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
rightrate=sum(y_pred==y_test['200'])/1385#计算精准度-90.39%正确率


test=pd.read_csv('pca-test-x.csv',index_col=0)
Y_pred=clf.predict(test)#这是预测结果，存储为perd_y.csv文件
```

