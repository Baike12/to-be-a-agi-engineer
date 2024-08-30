### stragety模式是什么
- 定义一些列算法，将每种算法放到独立的类中，使算法的对象能够互相替换
### 适用场景
- 需要使用对象中各种不同的算法变体，希望在运行时切换算法
### 实现
#### 概念
- 主体：维护指向具体策略的引用，只通过策略接口与对象交流
- 策略接口：声明一个上下文用来执行策略的方法
- 具体策略：实现上下文用的算法的各种变体
- 客户端：创建特定策略对象，并将策略对象传递给上下文，上下文踢动一个设置器让客户端运行时替换相关联的策略
#### 具体实现
- 缓存类与缓存移除算法解耦，以实现缓存类可以使用不同的算法
- 定义移除策略接口
- 定义不同的移除策略，这些策略都实现移除策略接口
- 定义cache：
	- 成员：移除算法、存储的数据、容量、最大容量
	- 方法：初始化一个cache、设置移除算法、添加数据、获取数据、移除数据
- 客户端：
	- 创建lfu、使用lfu初始化cache
	- 创建lru、设置lru为cache的移除算法、添加元素
	- 添加
#### c++实现
```c++

```
#### go实现
```go
package strategy

import (
	"fmt"
	// "github.com/go-ini/ini"
)

// 移除算法接口
type EvictionAlgo interface {
	evict(c *Cache)
}

// 具体策略
type Fifo struct {
}

// 实现接口
func (f *Fifo) evict(c *Cache) {
	fmt.Println("evicting with fifo")
}

type Lru struct {
}

func (lr *Lru) evict(c *Cache) {
	fmt.Println("evicting with lru")
}

// // 定义上下文
type Cache struct {
	storage      map[string]string
	evictionAlgo EvictionAlgo
	capacity     int
	maxCapacity  int
}

// 初始化cache
func initCache(e EvictionAlgo) *Cache {
	storage := make(map[string]string)
	return &Cache{
		storage:      storage,
		evictionAlgo: e,
		capacity:     0,
		maxCapacity:  2,
	}
}

// 设置cachede的算法
func (c *Cache) setEvictionAlgo(e EvictionAlgo) {
	c.evictionAlgo = e
}

// 添加元素
func (c *Cache) add(key, val string) {
	if c.capacity == c.maxCapacity {
		c.evict()
	}

	c.capacity++
	c.storage[key] = val
}

// 删除元素
func (c *Cache) evict() {
	c.evictionAlgo.evict(c) // 调用具体算法的移除操作
	c.capacity--
}

func Strategy() {
	fifo := &Fifo{}
	cache := initCache(fifo)

	cache.add("a", "1")
	cache.add("b", "2")
	cache.add("c", "3")

	lru := &Lru{}
	cache.setEvictionAlgo(lru)

	cache.add("d", "4")
}

```
### 扩展
- 添加策略算法就行