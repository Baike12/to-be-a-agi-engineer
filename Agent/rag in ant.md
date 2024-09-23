# 1	输入
- 重写用户输入
- embedding之后
# 2	向量检索挑战
## 2.1	存储成本
## 2.2	召回率和召回精度
- 提升精度带来的延时
- 索引近似性难以保证对未知查询的召回率
- 索引的静态性导致部分查询总是出错
## 2.3	复杂数据检索形态
# 3	实践
单一向量索引不可能三角
- 实时性
- 成本
- 高召回率
## 3.1	hnsw
都放在内存，实时性和召回率好
## 3.2	ivf-pq
量化，精度低
## 3.3	diskann
内存 + 磁盘，更新困难
向量数据库 = 存储 + 向量索引

# 4	vectorDB
蚂蚁自研
## 4.1	索引
### 4.1.1	内存
混合hnsw和diskann：
- 热点数据用hnsw，非热点用diskann
对于热点数据，先使用hnsw建立索引，同时在后台使用线程将这些索引用diskann重新创建索引，然后将hnsw替换
### 4.1.2	qps

### 4.1.3	算法
在第一阶段低精度的检索后不直接进行第二阶段的高精度检索，而是通过一个分类器






