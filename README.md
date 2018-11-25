# recommender

使用Octave实现的基于协同过滤的推荐系统算法。

## 使用方法

1. 将远程仓库克隆至本地：

   ```shell
   $ git clone https://github.com/sourcerunner/recommender.git
   ```

2. 在Octave中，切换工作目录到本地仓库：

   ```octave
   >> cd 'recommender'
   ```

3. 调节`exec.m`文件中的参数，然后执行命令：

   ```octave
   >> exec
   ```


## 数据集

使用电影评价数据集，数据来自[MovieLens](https://grouplens.org/datasets/movielens/)

> MovieLens 100K Dataset 和 MovieLens 1M Dataset

## 评测方法

- 数据集按照 randomly 80% training / 20% testing 划分

- 评测指标用RMSE。

## 模型

实现基于matrix factorization的协同过滤算法。

- [x] basic model
- [x] model with global mean rating mu
- [x] model with bias bi and bu
- [ ] additional input source
