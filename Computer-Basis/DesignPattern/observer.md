### observer模式是什么
- 允许定义一种订阅机制，在发布者发布消息时通知订阅者
### 适用场景
- 一个对象a需要关注对象b的状态时
### 实现
#### 概念
- 发布者：自身状态改变后通知订阅者，并提供新增和退出订阅的机制
- 发送者：事件发生时，遍历订阅者列表并调用每个订阅者的通知方法
- 订阅者接口：一般只实现一个update方法，用于发布者在事件发生时更新信息
- 具体订阅者：执行一些操作回应发布者的通知，所有订阅者都实现相同接口
- 客户端：分别创建订阅者和发布者
#### 具体实现
- 发布者接口：定义订阅方法、退订方法和通知方法
- 具体发布者：包含订阅者列表，发布者自己的名字，
- 观察者接口：包含一个更新方法，这个方法会被发布者调用来通知订阅者；还有一个获取id的接口
- 具体观察者：包含自己的id，实现观察者接口
#### c++实现
```c++

```
#### go实现
```go
package observer

import "fmt"

// 订阅者接口
type Observer interface {
	update(string)
	getID() string
}

// 发布者接口
type Subject interface {
	register(o Observer)
	deregister(o Observer)
	notifyAll()
}

// 具体订阅者
type Customer struct {
	id string
}

func (c *Customer) update(itemName string) {
	fmt.Printf("sending message %s to %s\n", itemName, c.id)
}

func (c *Customer) getID() string {
	return c.id
}

// 具体发布者
type Subjecter struct {
	observerList []Observer
	name         string
	inStock      bool
}

// 构造一个发布者
func newSubjecter(name string) *Subjecter {
	return &Subjecter{
		name: name,
	}
}

func (s *Subjecter) notifyNow() {
	fmt.Printf("we have %s now, you can buy it now\n", s.name)
	s.inStock = true
	s.notifyAll()
}

func (s *Subjecter) notifyAll() {
	for _, o := range s.observerList {
		o.update(s.name)
	}
}

func (s *Subjecter) register(o Observer) {
	s.observerList = append(s.observerList, o)
}

func (s *Subjecter) deregister(o Observer) {
	observerListLen := len(s.observerList)

	for i, ob := range s.observerList {
		if ob.getID() == o.getID() {
			s.observerList[observerListLen-1], s.observerList[i] = s.observerList[i], s.observerList[observerListLen-1]
			s.observerList = s.observerList[:observerListLen-1]
		}
	}
}

func ObserverFunc() {
	sub := newSubjecter("melon")

	ob1 := &Customer{id: "159"}
	ob2 := &Customer{id: "185"}

	sub.register(ob1)
	sub.register(ob2)

	sub.notifyNow()
}

```
### 扩展
- 