# Clustering Schoolwork描述文档

## 任务列表

1. 将鸢尾花数据集画成图的形式。
2. 确定一个合适的**阈值**，只有两个样本之间的相似度大于该阈值时，这两个样本之间才有一条边。
3. 求取带权**邻接矩阵**
4. 根据邻接矩阵进行聚类。
5. 将聚类结果可视化，重新转换成图的形式，其中每一个簇应该用一种形状表示，比如分别用圆圈、三角和矩阵表示各个簇。
6. 求得分簇正确率。

## step1 求带权临接矩阵

鸢尾花数据集是向量的形式而非图的形式，可以通过高斯核来计算向量之间的相似度。

![](https://github.com/m-L-0/18a-lidongliang--2016-449/blob/master/Clustering%20Schoolwork/image/%E9%AB%98%E6%96%AF%E6%A0%B8.png)

```python
#文件——图谱数据预处理.ipynb
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
data = load_iris()
x=data.data
y=data.target
#先算相似度矩阵
similarity=np.zeros([150,150])
for i in range(150):
    a=x[i]
    for j in range(i+1,150):
        b=x[j]
        A=a-b
        similarity[i][j]=similarity[j][i]=np.exp(-np.matmul(A,A)) 
```

## step2 确定阈值，化简邻接矩阵

阈值我们选择0.777，如果两个向量之间相似度低于这个阈值，就认为相似度为0，可以简化运算。

``` python 
#文件——图谱数据预处理.ipynb
ttt=0.777
similarity[similarity<ttt]=0
```

## step3 谱聚类

算法：

1. 如果数据集是向量的形式，首先计算其相似度矩阵作为邻接矩阵A，如果是图的形式，可以直接得到A。 
2. 由A得到度数矩阵和拉普拉斯矩阵。
3. 如果选择比例割，求拉普拉斯矩阵的特征值和特征向量，如果选择归一割，则求归一化后的对称或者不对称拉普拉斯矩阵的特征值和特征向量。 
4. 找到最小的k个特征值对应的特征向量组成矩阵U。 
5. 对U的每一行进行归一化，得到矩阵Y。 
6. 将Y的每一行看成是一个数据，以kmeans算法或其他快速聚类算法进行聚类。 

```python
#文件——图谱数据预处理.ipynb
#计算度数矩阵
Degree=np.zeros([150,150])
for i in range(150):
    Degree[i][i]=np.sum(similarity[i])
#计算拉普拉斯矩阵
Laplace=Degree-similarity
#用归一割
#计算非归一化对称拉普拉斯矩阵
Degree_sqrt=1/Degree
Degree_sqrt[Degree_sqrt==np.Inf]=0
toone_Laplace=np.matmul(Degree_sqrt,Laplace)
#求非归一化对称拉普拉斯矩阵的特征值和特征向量
eigenvalue,eigenvector=np.linalg.eig(toone_Laplace)
temp=pd.Series(eigenvalue)
#选择最小的k个特征值对应的特征向量
mineigvector=eigenvector[:,[5,6,7,8,1,31,35,34,20,0,19,149,147,148,146,145,144]]
#单位化
for i in range(150):
    T=np.sqrt(np.matmul(mineigvector[i],mineigvector[i]))
    mineigvector[i]=mineigvector[i]/T
#存储为文件
Clustering_data=pd.DataFrame(mineigvector)
Clustering_data.to_csv('Clustering_data.csv')
```

通过PCA降维，将处理结果可视化

```python
#文件——鸢尾花可视化.ipynb
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
iris = load_iris()
y = iris.target
X=pd.read_csv('Clustering_data.csv',index_col=0)
#降维
n_components = 2
pca = PCA(n_components=n_components)
X_transformed = pca.fit_transform(X)
#绘图
colors = ['navy', 'turquoise', 'darkorange']
title="PCA"
plt.figure(figsize=(8, 8))
for color, i, target_name in zip(colors, [0, 1, 2], iris.target_names):
    plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1],
                color=color, lw=2, label=target_name)
plt.legend(loc="best", shadow=False, scatterpoints=1)
plt.title(title + " of iris dataset")
#plt.axis([-4, 4, -1.5, 1.5])
plt.show()
d=pd.DataFrame(X_transformed)
d.to_csv('PCA-data.csv')
```

![处理结果](https://github.com/m-L-0/18a-lidongliang--2016-449/blob/master/Clustering%20Schoolwork/image/PAC-iris.png)

通过可视化可以看出我们的分簇效果比较理想

## step4 K-means聚类

算法：

创建k个点作为初始的质心点（***随机选择***）

当任意一个点的簇分配结果***发生改变***时

​	对数据集中的每一个数据点

​		对每一个质心

​			计算质心与数据点的距离

​		将数据点分配到距离***最近***的簇

​	对每一个簇，计算簇中所有点的***均值***，并将均值作为质心

```python
#文件——聚类.ipynb
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
#随机产生三个0-150的整数，返回列表
def randomindex():
    return [random.randint(51,150) for i in range(3)]
#穿入质心，完成分类
def cluster(conter):
    Distance=[]
    for i in range(150):
        temp=[]
        for j in range(3):
            a=np.subtract(data.iloc[i],conter.iloc[j])
            temp.append(np.matmul(a,a))
        Distance.append(temp)
    #分簇
    cluster=[]
    for i in range(150):
        cluster.append(Distance[i].index(min(Distance[i])))
    return np.array(cluster),Distance
#传入分簇结果，更新质心
def ChangeCenter(cluster):
    conter_next=np.zeros([3,2])
    conter_next[0]=np.sum(data[cluster==0])
    conter_next[1]=np.sum(data[cluster==1])
    conter_next[2]= np.sum(data[cluster==2])
    conter_next[0]/=np.sum(cluster==0) if np.sum(cluster==0)!=0 else 1
    conter_next[1]/=np.sum(cluster==1) if np.sum(cluster==1)!=0 else 1
    conter_next[2]/=np.sum(cluster==2) if np.sum(cluster==2)!=0 else 1
    return pd.DataFrame(conter_next)

data=pd.read_csv('PCA-data.csv',index_col=0)
index=randomindex()
#随机质心
conter=data.iloc[index]
#迭代更新质心
while True:
    cluster_frist,D1=cluster(conter)
    conter_next=ChangeCenter(cluster_frist)
    cluster_second,D2=cluster(conter_next)
    conter=conter_next
    if np.sum(cluster_frist==cluster_second)==150:
        break
#画图        
plt.figure(1)
plt.clf()
color=['r','b','g']
for i,c in zip(range(3),color):
    labels=cluster_frist==i
    plt.scatter(data.iloc[labels,0], data.iloc[labels,1], marker='o',c=c)
plt.show()        
```

![](https://github.com/m-L-0/18a-lidongliang--2016-449/blob/master/Clustering%20Schoolwork/image/%E8%81%9A%E7%B1%BB.png)

通过可视化可以得知自己写的K-means的分簇是有效的，结果比较理想。

## step5 计算准确率

根绝某一次分类结果：红色为“0”簇，蓝色为“1”簇，绿色为“2”簇

```python
num1=np.sum(cluster_frist[0:50]==0)
num2=np.sum(cluster_frist[100:150]==1)
num3=np.sum(cluster_frist[50:100]==2)
accuracy=(num1+num2+num3)/150
#准确率为81.33%
```

还有将近20%的数据分类错误是因为，K-means根据点到质心的距离进行分簇的，错分类的数据集里其他质心比较近。

## step6 根据邻接矩阵，将数据向量转化成图

1.利用sklearn.datasets.samples_generator.make_blobs生成三个簇。

```python
#文件——网络图.ipynb
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
X, y = make_blobs(n_samples=150, centers=[(1,14),(7.5,1),(14,14)], n_features=2,cluster_std=[3.0,3.0,3.0],random_state=0,shuffle=False)
x=np.array(X)
#字典pos储存150个点的坐标
pos={}
for i in range(1,151):
    pos[i]=x[i-1]
```

2.根据邻接矩阵，在networkx中绘制图。

```python
#文件——网络图.ipynb
data=pd.read_csv('similarity.csv',index_col=0)
data=np.array(data)
G = nx.Graph()
#参加节点
for i in range(1,151):
    G.add_node(i)
#参加边
for i in range(1,150):
    for j  in range(i,150):
        if data[i][j] != 0:
            print(i,j)
            G.add_edge(i,j,weight=data[i][j])
#按簇将节点区分         
nodes1=[i for i in range(1,51)]
nodes2=[i for i in range(51,101)]
nodes3=[i for i in range(101,151)]
#按权重将边区分
edge1 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.9274]
edge2 = [(u, v) for (u, v, d) in G.edges(data=True) if (d['weight'] < 0.9274) & (d['weight'] > 0.8)]
edge3 = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.8]
#标记点
nx.draw_networkx_nodes(G, pos, node_size=20,nodelist=nodes1,node_shape='s')
nx.draw_networkx_nodes(G, pos, node_size=20,nodelist=nodes2,node_shape='o')
nx.draw_networkx_nodes(G, pos, node_size=20,nodelist=nodes3,node_shape='>')
#标记边
nx.draw_networkx_edges(G, pos, edgelist=edge1,alpha=0.8,
                       width=2)
nx.draw_networkx_edges(G, pos, edgelist=edge2,
                       width=1, alpha=0.5, edge_color='gray')
nx.draw_networkx_edges(G, pos, edgelist=edge3,
                       width=1, alpha=0.5, edge_color='gray', style='dashed')
plt.show()
```

![](https://github.com/m-L-0/18a-lidongliang--2016-449/blob/master/Clustering%20Schoolwork/image/Figure_1.png)

## 

