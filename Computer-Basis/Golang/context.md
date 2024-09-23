# 1	Context
对goroutine组成的树形结构中对信号进行同步以减少计算资源浪费
## 1.1	特点
- 父context可以创建多个子context，子context一定指向某个父context
- 生命周期终止事件传递的单向性：如果父节点被终止了，所有子节点都收到消息被终止
## 1.2	用处
- 多个goroutine同时订阅上下文的Done管道的消息
## 1.3	如何用
- 数据存储
- 并发的协调控制
## 1.4	实现原理
```go
type Context interface {
	Deadline() (deadline time.Time, ok bool)
	Done() <-chan struct{}	
	Err() error
	Value(key any) any
}
```
- Done是一个只读的channel，在当前context被终止的时候会收到信号
# 2	emptyContext
实现了context接口但是什么都没干，所有context的基础
调用Background()的时候返回backgroundCtx，而backgroundCtx实际就是一个emptycontext
```go
func Background() Context {
	return backgroundCtx{}
}
type backgroundCtx struct{ emptyCtx }
```
# 3	cancelcontext
## 3.1	组成
```go
type cancelCtx struct {
	Context

	mu       sync.Mutex            // protects following fields
	done     atomic.Value          // of chan struct{}, created lazily, closed by first cancel call
	children map[canceler]struct{} // set to nil by the first cancel call
	err      error                 // set to non-nil by the first cancel call
	cause    error                 // set to non-nil by the first cancel call
}
```
- children：当前context的子context
```go
type canceler interface {
	cancel(removeFromParent bool, err, cause error)
	Done() <-chan struct{}
}
```
- 这里因为cancelctx自身只关注子context的cancel和done的情况，将这两个属性抽象出来，体现了低内聚
## 3.2	方法
### 3.2.1	创建子context
```go
func WithCancel(parent Context) (ctx Context, cancel CancelFunc) {
	c := withCancel(parent)
	return c, func() { c.cancel(true, Canceled, nil) }
}

func withCancel(parent Context) *cancelCtx {
	if parent == nil {
		panic("cannot create context from nil parent")
	}
	c := &cancelCtx{}
	c.propagateCancel(parent, c)
	return c
}
```
- 返回一个context和这个context的取消函数
- 父context必须不为空
- propagateCancel用于保证父亲被取消孩子都被取消
```go
func (c *cancelCtx) propagateCancel(parent Context, child canceler) {
	c.Context = parent

	done := parent.Done()
	if done == nil {
		return // parent is never canceled
	}

	select {
	case <-done:
		// parent is already canceled
		child.cancel(false, parent.Err(), Cause(parent))
		return
	default:
	}

	if p, ok := parentCancelCtx(parent); ok {
		// parent is a *cancelCtx, or derives from one.
		p.mu.Lock()
		if p.err != nil {
			// parent has already been canceled
			child.cancel(false, p.err, p.cause)
		} else {
			if p.children == nil {
				p.children = make(map[canceler]struct{})
			}
			p.children[child] = struct{}{}
		}
		p.mu.Unlock()
		return
	}

	if a, ok := parent.(afterFuncer); ok {
		// parent implements an AfterFunc method.
		c.mu.Lock()
		stop := a.AfterFunc(func() {
			child.cancel(false, parent.Err(), Cause(parent))
		})
		c.Context = stopCtx{
			Context: parent,
			stop:    stop,
		}
		c.mu.Unlock()
		return
	}

	goroutines.Add(1)
	go func() {
		select {
		case <-parent.Done():
			child.cancel(false, parent.Err(), Cause(parent))
		case <-child.Done():
		}
	}()
}
```
- 判断父上下文是否也是cancelctx，如果是而且父上下文没被取消直接将当前上下文加入到父上下文的children中
	- 如何判断：只有cancelctx调用Value方法传入cancelctxkey时才会返回它本身
	- 
```go
p, ok := parent.Value(&cancelCtxKey).(*cancelCtx)
	if !ok {
		return nil, false
	}
```
- 如果不是cancelctx，用一个协程监听父上下文是否取消
### 3.2.2	取消
主要做三件事：
- 设置导致cancel的原因
- 让上游使用Done方法时没法使用，通过关闭channel和赋值一个标识channel被关闭的变量
- 取消所有孩子
```go
func (c *cancelCtx) cancel(removeFromParent bool, err, cause error) {
	if err == nil {
		panic("context: internal error: missing cancel error")
	}
	if cause == nil {
		cause = err
	}
	c.mu.Lock()
	if c.err != nil {
		c.mu.Unlock()
		return // already canceled
	}
	c.err = err
	c.cause = cause
	d, _ := c.done.Load().(chan struct{})
	if d == nil {
		c.done.Store(closedchan)
	} else {
		close(d)
	}
	for child := range c.children {
		// NOTE: acquiring the child's lock while holding parent's lock.
		child.cancel(false, err, cause)
	}
	c.children = nil
	c.mu.Unlock()

	if removeFromParent {
		removeChild(c.Context, c)
	}
}
```
# 4	timeCtx
- 一个带有计时器和过期时间的cancelctx
```go
type timerCtx struct {
	cancelCtx
	timer *time.Timer // Under cancelCtx.mu.

	deadline time.Time
}
```
使用WithTimeout和WithDeadline创建timeCtx，最终都是调用WithDeadlineCause
### 4.1.1	WithDeadlineCause
主要逻辑是使用time.AfterFunc创建一个定时器，AfterFunc就是对cancel函数的调用
```go
		c.timer = time.AfterFunc(dur, func() {
			c.cancel(true, DeadlineExceeded, cause)
		})
```

# 5	valuectx
只有一对kv
两次放入相同值的key，会生成两个valuectx，所以如果有key相同的valuectx，获取到的值取决于开始获取的位置
```go
type valueCtx struct {
	Context
	key, val any
}
```
### 5.1.1	获取value
```go
func (c *valueCtx) Value(key any) any {
	if c.key == key {
		return c.val
	}
	return value(c.Context, key)
}
```
如果获取不到，就到父节点去获取
```go
func value(c Context, key any) any {
	for {
		switch ctx := c.(type) {
		case *valueCtx:
			if key == ctx.key {
				return ctx.val
			}
			c = ctx.Context
		case *cancelCtx:
			if key == &cancelCtxKey {
				return c
			}
			c = ctx.Context
		case withoutCancelCtx:
			if key == &cancelCtxKey {
				// This implements Cause(ctx) == nil
				// when ctx is created using WithoutCancel.
				return nil
			}
			c = ctx.c
		case *timerCtx:
			if key == &cancelCtxKey {
				return &ctx.cancelCtx
			}
			c = ctx.Context
		case backgroundCtx, todoCtx:
			return nil
		default:
			return c.Value(key)
		}
	}
}
```