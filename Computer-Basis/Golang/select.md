### select
- 让协程阻塞着等待多个channel可读或者可写
- 多个case同时触发，随机执行一个
#### 现象
##### select在channel上非阻塞的收发
- 使用default
- 用于：有时候只想知道事件有没有发生，不关心发生了什么，比如错误
##### 随机执行
- 防止饥饿
#### 数据结构
```go
// Select case descriptor.
// Known to compiler.
// Changes here must also be made in src/cmd/compile/internal/walk/select.go's scasetype.
type scase struct {
	c    *hchan         // chan
	elem unsafe.Pointer // data element
}
```
#### 实现原理
##### 空select
- 直接阻塞：调用block然后调用gopark，让出cpu使用，并永远不会被唤醒
##### 单一case
- 直接对chan是否为空进行判断
##### 两个case一个是default
发送
- 因为有default，所以就算缓冲区不足或者接收方不存在都不会阻塞当前goroutine
##### 三个及以上case的常见流程
- 将case转换成scase结构体
- 使用selectgo选择一个scase结构体
- 执行选中case
选择函数selectgo：
- 确定两种选择顺序：
	- 随机轮询 
	- 加锁顺序：根据cahnnel地址排序确定加锁顺序
- 锁定channel后进入循环
	- 查找准备就绪的channel
	- 将当前goroutine加入channel对应的队列上等待其他goroutine唤醒
	- 当前goroutine被唤醒之后找到满足条件的channel并处理
异常
- 从关闭的channel中接收数据会清除cahnnel
- 向关闭的channel发送数据会panic