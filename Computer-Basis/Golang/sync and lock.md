## 锁与同步原语
### 互斥锁
```go

```
### 读写锁
- 获取写锁会先阻塞写锁的获取，然后才阻塞读锁获取，避免读饿死
#### 获取写锁
```go
func (rw *RWMutex) Lock() {
	rw.w.Lock()
	r := atomic.AddInt32(&rw.readerCount, -rwmutexMaxReaders) + rwmutexMaxReaders
	if r != 0 && atomic.AddInt32(&rw.readerWait, r) != 0 {
		runtime_SemacquireMutex(&rw.writerSem, false, 0)
	}
}
```
- 先获取互斥锁，防止其他协程来获取写锁
- 将读者数量  - 最大读者数量，防止其他协程来获取读锁
- r是当前读者数量，如果r不等于0，将读等待数量原子的增加到当前读者数量，这些就是阻塞当前写操作的读者
- 用于获取写信号量
#### 释放写锁
```go
func (rw *RWMutex) Unlock() {
	r := atomic.AddInt32(&rw.readerCount, rwmutexMaxReaders)
	if r >= rwmutexMaxReaders {
		throw("sync: Unlock of unlocked RWMutex")
	}
	for i := 0; i < int(r); i++ {
		runtime_Semrelease(&rw.readerSem, false, 0)
	}
	rw.w.Unlock()
}
```
- 恢复读者数量，因为获取写锁的时候把读者数量减掉了，现在释放写锁要加回来，如果恢复了发现最大读者数量 > =最大度读者数量，说明这个锁之前被释放了，报错
- 通过信号量释放所有因为写操作阻塞的读协程
#### 获取读锁
```go
func (rw *RWMutex) RLock() {
	if atomic.AddInt32(&rw.readerCount, 1) < 0 {
		runtime_SemacquireMutex(&rw.readerSem, false, 0)
	}
}
```
- 将读者数量 + 1，如果还 < 0就说明有其他协程获得了写锁，休眠自己
#### 释放读锁
```go
func (rw *RWMutex) rUnlockSlow(r int32) {
	if r+1 == 0 || r+1 == -rwmutexMaxReaders {
		throw("sync: RUnlock of unlocked RWMutex")
	}
	if atomic.AddInt32(&rw.readerWait, -1) == 0 {
		runtime_Semrelease(&rw.writerSem, false, 1)
	}
}
```
-  将读者数量 - 1，如果 < 0，说明有写锁在试图获取，因为在获取杜所的时候将读数量 + 1了的，这里 - 1不会 < 0， < 0说明是写锁干的；当然，不 < 0直接解锁成功， < 0就调用慢解锁
```go
func (rw *RWMutex) rUnlockSlow(r int32) {
	if r+1 == 0 || r+1 == -rwmutexMaxReaders {
		throw("sync: RUnlock of unlocked RWMutex")
	}
	if atomic.AddInt32(&rw.readerWait, -1) == 0 {
		runtime_Semrelease(&rw.writerSem, false, 1)
	}
}
```
- 如果读者数量 + 1 = 0，说明读者读锁被释放过了，这是上一步将它 - 1造成的，和写无关
- 如果 + 1 = 负的最大读者数量，也说明被释放了，不过和写相关
- 将写操作的读者数量 - 1，如果 - 1之后发现没有读者了，就通知写协程来写
### Cond
让一组goroutine在满足特定条件时被唤醒
```go
type Cond struct {
	noCopy  noCopy
	L       Locker
	notify  notifyList
	checker copyChecker
}
```
- 主要就是notifylist，是一个协程列表
提供了两个方法唤醒休眠协程
- signal：唤醒最前面的
- broacast：唤醒队列中所有协程
提供了一个wait来等待唤醒，但是调用wait之前要先获取锁
#### 使用场景
- 比for循环等待条件好，可以让出cpu

## 扩展原语
#### errgroup
- 捕获一个错误
#### weighted
- 带权重的信号量
提供了四个方法：
- 初始化
- 阻塞的获取资源
- 非阻塞的获取资源，没有资源返回false
- 释放资源
#### singleflight
可以减少相同请求对下游的压力
提供了两个方法：
- Do：同步等待
- DoChan：返回channel异步等待，就是说返回的是一个channel，然后从cahnnel中获取
还有一个删除方法
- forget用来删除group中的一个key