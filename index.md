# 天池龙珠计划 机器学习训练营 day1
## 机器学习算法（一）：基于逻辑回归的分类预测
### 1逻辑回归的介绍和应用
#### 1.1逻辑回归的介绍
逻辑回归（logistic regression,LR)虽然名中带有回归两个字．但逻辑回归具实一个分类模型，并且广泛应用于各个领域之中。

虽然现在深度学习相对于这些传统方法更为火热，但实则这些传统方法由于具独特的优势依然广泛应用于各个领域中。

而对于逻辑回归，最为突出的两点就是其模型简单和模型的可解释性强。

回归模型的优劣势：

- 优点：实现简单，易于理解和实现；计算代价不高，速度很快，存储资源低；

- 缺点：容易欠似合，分类精度可能不高。

1．1逻辑回归的应用

逻辑回归模型广泛用于各个领域，包括机器学习，大多数医学领域和社会科学。也是一个理解数据的好工具。

但同时由于其本质上是一个线性的分类器所以不能应对较为复杂的数据情况。

很多时候我们也会拿逻辑回归模型去做一些任务尝试的基线（基础水平）。

  转自[天池](https://dsw-dev.data.aliyun.com/?spm=5176.20222472.J_3678908510.1.1a0b67c2UeRr4Y#/?fileUrl=http://tianchi-media.oss-cn-beijing.aliyuncs.com/DSW/1back/back.ipynb&fileName=back.ipynb)

---

# 天池龙珠计划 机器学习训练营 day2
### 2逻辑回归Demo实践
#### 2.1逻辑回归代码流程
1. 函数库导入
2. 模型训练
3. 模型参数查看
4. 数据和模型可视化
5. 模型预测
#### 2.2逻辑回归python代码实践
1. 函数库导入

`## 基础函数库`

`import numpy as np`

`## 导入画图库`

`import matplotlib.pyplot as plt`

`import seaborn as sns`

`## 导入逻辑回归模型函数`

`from sklearn.linear_model import LogisticRegression`

2. 模型训练

`x_features = np.array([[-1,-2],[-2,-1],[-3,-2],[1,3],[2,1],[3,2]])`

`y_label = np.array([0,0,0,1,1,1])`

`lr_clf = LogisticRegression()`

`lr_clf = lr_clf.fit(x_features,y_label) #其拟合方程为 y=w0+w1*x1+w2*x2`

3. 模型参数查看

`print('the weight of Logistic Regression:',lr_clf.coef_)`

`print('the intercept(w0) of Logistic Resgression:',lr_clf.intercept_)`

Logistic回归的权重：[[0.73455784 0.69539712]]

Logistic回归的截距（w0）：[-0.13139986]

4. 数据和模型可视化

`plt.figure()`

`plt.scatter(x_features[:,0],x_features[:,1],c=y_label,s=50,cmap='viridis')`

`plt.title('Dataset')`

`plt.show()`

5. 模型预测

`x_features_new1 = np.array([[0,-1]])`

`x_features_new2 = np.array([[1,2]])`

`y_label_new1_predict = lr_clf.predict(x_features_new1)`

`y_label_new2_predict = lr_clf.predict(x_features_new2)`

`print('The New point 1 predict class:\n',y_label_new1_predict)`

`print('The New point 2 predict class:\n',y_label_new2_predict)`

`y_label_new1_predict_proba = lr_clf.predict_proba(x_features_new1)`

`y_label_new2_predict_proba = lr_clf.predict_proba(x_features_new2)`

`print('The New point 1 predict Probability of each class:\n',y_label_new1_predict_proba)`

`print('The New point 2 predict Probability of each class:\n',y_label_new2_predict_proba)`

新的点1预测类：[0]

新的点2预测类：[1]

新的第1点预测为每一类的概率：[[0.69567724 0.30432276]]

新的第2点预测为每一类的概率：[[0.11983936 0.88016064]]

# 天池龙珠计划 机器学习训练营 day3
### 3基于鸢尾花（Iris）数据集的逻辑回归分类实践
#### 3.1逻辑回归代码流程
1. 库函数导入

`import numpy as np  ##科学计算基础包`

`import pandas as pd  ##数据分析处理包`

`import matplotlib.pyplot as plt  ##绘图包`

`import seaborn as sns  ##绘图包`

数据集内容：

1. 花萼长度 sepal length
2. 花萼宽度 sepal width
3. 花瓣长度 petal length
4. 花瓣宽度 petal width
5. 分类标签（0,1，2）

2. 数据读取

`from sklearn.datasets import load_iris`

`data = load_iris()`

`iris_target = data.target`

`iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)`

3. 数据信息简单查看

`iris_features.info()`

`iris_features.head()`

`iris_features.tail()`

`iris_target`

`pd.Series(iris_target).value_counts() ##查看每个类别数量`

`iris_features.describe() ##对特征的统计描述`

4. 可视化描述

合并标签和特征信息

`iris_all = iris_features.copy()`

`iris_all['target'] = iris_target`

特征与标签组合的散点可视化

`sns.pairplot(data=iris_all,diag_kind='hist', hue='target')`

`plt.show()`

`for col in iris_features.columns:`

`  sns.boxplot(x='target', y=col, saturation=0.5, palette='pastel', data=iris_all) ##箱型图`

`  plt.title(col)`
`  plt.show()`

![pic](https://github.com/oumiga12/oumiga12.GitHub.io/blob/main/3Dscatter.png)

<img src="https://github.com/oumiga12/oumiga12.GitHub.io/blob/main/3Dscatter.png" width="%50" height="%50" />

![pic](https://github.com/oumiga12/oumiga12.GitHub.io/blob/main/3Dscatter1.png)

<img src="https://github.com/oumiga12/oumiga12.GitHub.io/blob/main/3Dscatter1.png" width="%50" height="%50" />


















