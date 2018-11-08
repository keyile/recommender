# recommender

使用Octave实现的基于协同过滤的推荐系统算法。

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