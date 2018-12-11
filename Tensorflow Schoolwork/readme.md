# TensorFlow Schoolwork代码描述文档

首先先介绍一下knn的基本原理：

KNN是通过计算不同特征值之间的距离进行分类。

整体的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，则该样本也属于这个类别。

KNN算法要解决的核心问题是K值选择，它会直接影响分类结果。

## K近邻实现

### 准备

```python
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn import model_selection
import numpy as np
#开启eager模式
tf.enable_eager_execution()
#导入数据
data = load_iris()
x=data.data
y=data.target
#分割数据集
X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, shuffle=True)
```

### 函数

```python
#求决策点到样本的距离
#计算测试集第i个样本到训练集的L2距离
#返回测试集第i个样本到训练集的L2距离的darray
def Distance(i):
    A=X_train-X_test[i]
    TensorDistance=tf.reduce_sum(tf.multiply(A,A),1)
    distance=[*TensorDistance.numpy()]
    distance[i]=np.inf
    return distance
#找到K个近的索引
#传入你要找第i个点的K个近邻
#返回K个近邻的索引列表
def KNeighboring(i,K):
    index=[]
    distance=Distance(i)
    for i in range(K):
        temp=distance.index(min(distance))
        index.append(temp)
        distance[temp]=np.inf
    return index
#胜者为王
#传入K个近邻的索引
#返回决策结果
def Decision(index):
    labels=y_train[index]
    Set=set(labels.tolist())
    result={}
    for i in Set:
        result[i]=np.sum(labels==i)
    maxvalue=max(result.values())
    for i in Set:
        if result[i]==maxvalue:
            return i
```

### 寻找最优K值

```python
#寻找最优K值
#我们让K的取值为1-20
K=[i for i in range(1,21)]
Accuracy=[]
for k in K:
    Result=[]
    for i in range(len(X_test)):
        index=KNeighboring(i,k)
        result=Decision(index)
        Result.append(result)
    print(Result)
    right=tf.reduce_sum((y_test==Result).astype(int),0).numpy()
    print(right)
    accuracy=right/len(X_test)
    Accuracy.append(accuracy)
```

### 结果可视化

```python
#可视化
%matplotlib
import matplotlib.pyplot as plt
plt.plot(K, Accuracy,'r',marker='o')  #关键句,前两个参数是X、Y轴数据,其他参数指定曲线属性，如标签label，颜色color,线宽linewidth或lw,点标记marker
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy of K')
plt.locator_params('x',nbins=40)
plt.locator_params('y',nbins=20)
plt.show()
```

![](https://github.com/m-L-0/18b-lidongliang-16-449/blob/master/Tensorflow%20Schoolwork/image/Figure_1.png)

## 结论

尽可能避免偶然性，我们选择较大的K值，我们选择最优K值为19