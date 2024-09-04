### Context
对goroutine组成的树形结构中对信号进行同步以减少计算资源浪费
#### 实现原理
- 多个goroutine同时订阅上下文的Done管道的消息