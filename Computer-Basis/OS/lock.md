### 死锁
#### 条件
- 互斥
- 不可剥夺
- 持有并等待
- 环路等待
#### 解决方法
- 资源有序分配
### 锁
#### 互斥锁
- 获取不到锁会休眠，切换到其他线程，由操作系统执行，所以上下文切换代价大
#### 自旋锁
- 通过CAS函数在用户空间加解锁
#### 读写锁
- 写锁没被持有，允许多个线程持有读锁
- 写锁被持有，读写锁都被阻塞
- 读优先：有线程A在读，写锁B被阻塞，读锁C可以获取；会造成写饥饿
- 读优先：线程A在读，写锁B被阻塞，读锁C被阻塞；造成读饥饿
- 公平读写：用一个队列，先到先服务
#### 乐观锁与悲观锁
- 乐观锁：先无锁修改共享资源，再判断这段时间有没有冲突，一般通过版本号判断，有冲突就回退这次修改；适用于冲突概率小的情形