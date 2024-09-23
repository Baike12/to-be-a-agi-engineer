# 1	事务并行的问题
## 1.1	脏读
一个事务读到了另一个事务没提交的数据。
## 1.2	不可重复读
一个事务多次读取同一条数据，但是前后读到的数据不一样。
## 1.3	幻读
一个事务查询符合条件的数据条数，前后读到的条数不一样

# 2	隔离级别
## 2.1	读未提交
直接读取。
## 2.2	读已提交
每条语句执行时创建readview。
## 2.3	可重复读
在每个事务开启时创建readview。
mysql默认级别，通过两种方法在一定程度上避免了幻读：
### 2.3.1	快照读
普通select语句就是快照读，通过MVCC解决幻读
### 2.3.2	当前读
select for update，通过记录锁 + 间隙锁解决幻读
## 2.4	串行化
加锁避免并行访问

# 3	readview
![[Pasted image 20240918162026.png]]
- creator_trx_id：创建readview的事务id
- m_ids：创建readview时活跃且未提交的事务id列表
- min_trx_id：创建readview时最小的事务id
- max_trx_id：创建readview时要分配给下一个事务的id
在一行记录存储时有两个隐藏列
![[Pasted image 20240918162446.png]]
- trx_id：最近修改行记录的事务id
- roll_pointer：每次修改行记录后，旧版本被写到undo日志中，这个指针就是指向undo旧版本的
## 3.1	隔离原理
事务访问记录时有三种情况：
### 3.1.1	trx_id < min_trx_id
说明在创建readview前该记录就被修改了，所以对当前事务可见
### 3.1.2	trx_id >= max_trx_id
说明记录在创建readview之后才被修改的，所以记录对当前事务不可见
### 3.1.3	trx_id在min_trx_id和max_trx_id之间
这时要判断trx_id是否在m_ids中：
- 不在，说明记录虽然是在readview创建时正在被修改中，但是此时已经修改它的事务已经提交了，所以记录对当前事务可见
- 在，记录还在被修改中，所以不可见

**读已提交和可重复读的区别就是使用的readview不同，读已提交使用的是当前语句执行时创建的，可重复读使用的是事务开始时创建的。**

# 4	可重复读不能完全避免幻读
以下场景依然会出现幻读
- 事务a开启
- 事务b插入一条新数据，然后提交
- 事务a更新这条新数据，可以更新，虽然a不能select到新数据，但是可以update
- 事务a再查，就能查到新数据了
事务a：
![[Pasted image 20240918165336.png]]
事务b
![[Pasted image 20240918165354.png]]