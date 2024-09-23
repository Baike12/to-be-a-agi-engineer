# 1	undo log
存储引擎生成，记录每一次更新产生一条
## 1.1	作用
- 回滚
- mvcc
## 1.2	构成
![[Pasted image 20240919231033.png]]
- trx_id：修改当前记录的事务id
- roll_pointer：指向上一个版本的记录，构成版本链
# 2	redo log
- 也是由存储引擎生成。
## 2.1	内容
记录更新的时候，生成redo log，记录的是对数据页的什么位置进行了什么。
## 2.2	作用
用于前滚：事务提交之后数据库或者系统崩溃，使用redo log恢复
## 2.3	为什么需要redo log？直接记录数据不是更好？
因为redo log是顺序写，比记录数据随机写快得多
## 2.4	落盘机制
redo log也不是事务一提交就落盘的，会先写到redo log buffer中这个buffer默认16m大小，可以通过innodb_log_buffer_size配置，在一些时机下落盘：
- mysql关闭
- redo log buffer大小到达容量的一半
- 每隔1s
- 可以配置为：每次事务提交时落盘，由innodb_flush_log_at_trx_commit控制
innodb_flush_log_at_trx_commit配置：
- 0：事务提交时，redolog还是留在redo log buffer中，后台线程每隔1s会将redolog buffer中的数据落盘
- 1：事务提交时，直接落盘
- 2：事务提交时，写到内核的page cache中，后台线程每隔1s将page cache中的数据落盘
## 2.5	结构
由一个重做日志组构成，buffer pool里面有两个重做日志，ib_logfile0和ib_logfile1，每个1G大（可配置），两个日志文件构成一个环，对于已经落盘的日志要清理，用两个指针write pos和check point指向这个环的写入位置和清理位置，如果write pos赶上了check point，说明环满了，sql执行会被阻塞。
# 3	binlog
由server层生成。
## 3.1	内容
记录修改和库表结构变更
### 3.1.1	内容格式
- statement：记录的是sql，对于动态函数比如插入时间会导致主从不一致
- row：记录记录被修改成什么样了，所以记录量比较大
- mixed：自动使用上面两种
## 3.2	用途
备份恢复、主从复制
### 3.2.1	主从复制
流程：
- 主库先写binlog，再提交事务，然后回客户端响应
- 从库使用io线程连接主库的log dump线程，把binlog日志写到relay log中继日志中，然后给主库恢复同步成功相应
- 从库创建回放binlog的线程从relay log回放数据
类型：
- 同步复制：等主库和所有从库同步后才给客户端回响应
- 异步复制：主库自己提交成功就给客户端回响应
- 半同步辅助：等一部分从库同步成功就回响应
## 3.3	落盘机制
每个线程有一个binlog cache，用binlog_cache_size配置大小，超过这个大小就落盘。
还提供了sync_binlog来控制落盘：
- 0：每次事务提交只写到内核的page cache中，不落盘，由系统决定何时落盘
- 1：每次提交都立马落盘
- n：每次提交都写到page cache中，但是n个事务才落盘
# 4	两阶段提交
主要是因为redo log和binlog的写入不是同步的，可能出现一个成功一个失败的情况：
- redo log写入成功、binlog还没写入时宕机：恢复后主库有的数据从库没有
- binlog写入成功，redo log还没写入时宕机：恢复后从库有的数据主库没有
## 4.1	流程
- prepare阶段：将内部xa事务xid写入redo log，将redo log对应的事务状态设置为prepare，然后将redo log写入磁盘
- commit阶段：将xid写入binlog，然后将binlog持久化到磁盘，再调用存储引擎提交事务接口，将redolog状态设置为commit，这个commit状态的redolog不用写到磁盘，只要写到page cache中就行，因为binlog和prepare状态的redolog都写到磁盘了，事务可以成功提交了
![[Pasted image 20240920101521.png]]
## 4.2	异常分析
宕机重启后redo log会扫描xid，用xid去binlog中找：
- 发现没找到，就会回滚这个xid对应的事务
- 找到，提交这个xid对应的事务
两阶段提交以binlog写入作为事务提交成功标识，写日志会先redo log。
## 4.3	缺点：
io次数多：binlog和redo log都需要刷盘一次
锁竞争：多事务时保证事务日志按顺序写入需要用锁控制
## 4.4	组提交
prepare阶段不变
commit阶段分为三个阶段：
- flush：多个事务按顺序将日志从binlog写入page cache
- sync：将page cache中的日志落盘
- commit：各事务按顺序commit
三个阶段都由一个队列 + 锁维护，保证顺序。

## 4.5	磁盘io很高的缓解方法
- 组提交相关：binlog_group_commit_sync_delay和binlog_group_commit_sync_no_delay_count设置组提交刷盘等待时间和事务数量
- binlog相关：sync_binlog设置binlog在累计多少个事务之后才刷盘
- redo log相关：innodb_flush_log_at_trx_commit设置redo log刷盘时机

# 5	buffer pool
页为单位，大小为16k
## 5.1	种类
数据页，索引页，插入缓存页，undo页，等。。。。
### 5.1.1	undo页
记录更新前，生成undo log写入buffer pool的undo页
