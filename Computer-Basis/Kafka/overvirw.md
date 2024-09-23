# 1	topic
## 1.1	分区
保证集群的线性扩展
## 1.2	高可用
一个分区有多个副本，replication-factor设置副本数
### 1.2.1	角色
- leader：负责对外服务，维护一个ISR数组：保存副本编号，如果副本同步差太多会从数组剔除
- follower：负责备份

# 2	消息存放
## 2.1	键值对
### 2.1.1	发送
- 不指定key，kafka以轮询的方式将数据均匀写到不同partition中
- 指定key，相同key消息会放到同一个partition
# 3	Broker
- 一个broker就是一个服务
## 3.1	结构
包括一个leader和多个follower，但是leader的partition和leader的未必是一个
![[Pasted image 20240922000151.png]]
一号broker负责：
- partition1的读写请求
- partition0、partition2的从他们的leader同步
# 4	搭建伪分布式环境
## 4.1	创建步骤
- 拷贝多个配置文件到etc目录下，分别命名
- 修改配置文件
## 4.2	操作
### 4.2.1	创建主题
```shell
./kafka-topics.sh --bootstrap-server localhost:9091 --create --topic baike --partitions 3 --replication-factor 2

./kafka-topics.sh --bootstrap-server localhost:9091 --describe --topic baike
```
![[Pasted image 20240922010259.png]]
### 4.2.2	发送消息
```shell
./kafka-console-producer.sh --broker-list localhost:9091 --topic baike
```
### 4.2.3	消费消息
```shell
./kafka-console-consumer.sh --bootstrap-server localhost:9091 --topic baike
```
## 4.3	网络配置
### 4.3.1	listeners：kafka监听的端口
![[Pasted image 20240922095550.png]]
- 0.0.0.0：监听所有网卡的9092端口
### 4.3.2	advertised.listeners：对外宣称的地址，用于客户端连接，注册到zk中
![[Pasted image 20240922095706.png]]
- 客户端通过advertised.listeners访问
默认情况下advertised.listeners不用配置，使用listeners相同配置
![[Pasted image 20240922095847.png]]
- advertised.listeners不配置的情况下，自动推断为listeners的ip，然后给客户端连接用
上面只适合在同一网络或者ip可以互相访问的情况
#### 4.3.2.1	kafka部署在阿里云
- kafka服务端之间可以互相访问，但是外网不能直接访问
![[Pasted image 20240922100232.png]]
 所以要分为内网INTERNAL和外网EXTERNAL：
- listeners内网配置为9092，外网配置为9093，这里可以用0.0.0.0表示所有网卡
- advertised.listeners内网配置为9092，外网一定要配置为公网ip，不能是0.0.0.0
- 客户端可以根据处于内网还是外网访问不同的端口
 ![[Pasted image 20240922100331.png]]
 示例：
 ![[Pasted image 20240922100836.png]]
# 5	使用docker部署

# 6	消费模型
- 一个消费者可以消费多个分区
- 一个分区可以被多个消费者组中的消费者消费，但是一个分区不能被一个消费者组中的多个消费者消费
## 6.1	消费模式
### 6.1.1	发布订阅模式
每个消费者属于一个组，每个消息都需要被所有消费者组消费
### 6.1.2	一对一
将所有消费者放到一个组中，这样一个消息只被消费一次
## 6.2	消息顺序
### 6.2.1	生产者
同一个生产者发送到同一个分区的消息，先发的offset一定 < 后发的
同一个生产者发送到不同分区，消息顺序无法保证
### 6.2.2	消费者
只能保证一个分区的消息被顺序消费
### 6.2.3	顺序消费方案
- 只设置一个分区，那就没有扩展性和性能了
- 相同key会发送到一个分区，可以给需要按顺序消费的数据设置相同的key

# 7	消息传递语义
## 7.1	最少一次
- 生产者没收到响应就重试
- 消费者先读取消息，再提交消费位置，这样消费位置提交失败就会再读取一次相同消费位置的数据
- 这里的消费位置放在一个特殊的topic中，这个topic存放每一个消费者的消费位置
## 7.2	最多一次
- 生产者没收到响应不重试
- 先提交消费位置，再读取消息
## 7.3	精确一次
### 7.3.1	生产者
参数：
- enable.idempotence=true
- acts=all
### 7.3.2	消费者
加一个唯一id
# 8	生产者api
## 8.1	异步发送
先将消息放到生产者自己的发送缓冲区中，然后后台线程再发送给broker
## 8.2	同步发送
收到发送成功响应再发送下一条
## 8.3	批量发送
可以设置达到以下条件时发送
- 消息条数达到一个值
- 每个一段时间发送一次
## 8.4	ack
服务端给生产者的收到确认
- 0：生产者不等待服务端响应，放到缓冲区就认为发送成功
- 1：被leader存入本地
- all：被所有节点存入本地
# 9	消费者api
有一个topic：__consumer_offsets 保存消费者消费哪个主题哪个分区的哪个位置

自动提交用来实现最多一次
手动提交实现最少一次
# 10	事务
## 10.1	隔离级别
### 10.1.1	读未提交
导致脏读
### 10.1.2	读已提交
只能消费已经提交的数据

# 11	序列化
对象转换为二进制
## 11.1	序列化类型
9种：byte数组序列化等。。。
![[Pasted image 20240922110557.png]]
类似于grpc
# 12	record header
为了保证消费的顺序性，将所有消息使用相同的key，这样所有的消息都会放到同一个topic的同一个分区。但是因为不同的消息格式不一样，导致序列化有问题
## 12.1	schema registry
- 发送者在发送消息前将消息的schema发送到sr去获取一个id，将这个id加上消息一起发送给broker
- 消费者消费消息时用id去获取schema，然后根据schema反序列化
### 12.1.1	缺点
- 破坏了分布式
- 破坏了消息本身，因为要加上id