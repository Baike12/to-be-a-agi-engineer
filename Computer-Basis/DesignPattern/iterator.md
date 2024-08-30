### iterator模式是什么
- 在不暴露底层表现形式的情况下遍历集合中的所有元素
### 适用场景
- 减少遍历代码
- 需要对客户端隐藏底层数据结构的复杂性
- 需要遍历无法预知的数据结构
### 实现
#### 概念
- 迭代器：声明遍历集合的操作：获取下一个元素、获取当前位置、重新开始迭代等
- 具体迭代器
- 集合
- 具体集合
- 客户端
#### 具体实现
- 定义集合接口
- 定义具体集合，具体集合返回一个迭代器接口，所以可以返回后面实现了迭代器接口的具体迭代器
- 定义迭代器接口
- 定义具体迭代器，实现对集合的访问
- 注意：迭代器中要有数据集中的数据
#### c++实现
```c++

```
#### go实现
```go
package iterator

import "fmt"

// 定义集合接口，包含返回迭代器接口的方法
type Collection interface {
	createIterator() Iterator
}

// 定义迭代器接口，包含判断下一个元素是否存在、获取下一个元素两个方法
type Iterator interface {
	hasNext() bool
	getNext() *User
}

// 定义一个元素空接口
// type Elem interface{}

// 定义具体元素
type User struct {
	name string
	age  int
}

// 定义具体迭代器，包含下标和具体元素的指针数组
type UserIterator struct {
	index int
	users []*User
}

func (ui *UserIterator) hasNext() bool {
	if ui.index < len(ui.users) {
		return true
	}
	return false
}

func (ui *UserIterator) getNext() *User {
	if ui.hasNext() {
		user := ui.users[ui.index]
		ui.index++
		return user
	}
	return nil
}

// 定义具体集合
type UserCollection struct {
	users []*User
}

func (u *UserCollection) createIterator() Iterator {
	return &UserIterator{
		users: u.users,
	}
}

func IteratorFunc() {
	u1 := &User{
		name: "u1",
		age:  12,
	}

	u2 := &User{
		name: "u2",
		age:  24,
	}

	userCollection := &UserCollection{
		users: []*User{u1, u2},
	}

	iterator := userCollection.createIterator()

	for iterator.hasNext() {
		user := iterator.getNext()
		fmt.Printf("user name %s, user age %d", user.name, user.age)
	}
}

```
### 扩展
- 元素本身最好也是接口，最好是空接口