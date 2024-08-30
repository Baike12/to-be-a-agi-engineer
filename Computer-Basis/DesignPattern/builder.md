### builder是什么
- 分步骤创建复杂对象，使用相同创建代码生成不同类型和形式的对象
- 将构造代码放到生成器，并且不允许其他对象访问创建中的产品
- 主管类指定步骤顺序
### 适用场景
- 构造函数的参数太多，使用生成器可以只使用功能必要的步骤
- 可以创建不同形式的产品
- 构造组合树或者复杂对象
### 实现
- 定义产品
- 定义生成器接口
- 定义具体生成器，并实现接口
- 定义主管：
	- 主管类接受生成器接口作为参数
	- 在主管中定义需要的步骤
- 定义获取生成器的接口给客户端调用
- 使用
	- 创建一个生成器：这是一个未赋值的生成器实例
	- 把生成器作为参数创建一个主管实例
	- 由主管实例来对生成器赋值
	- 生成器有返回产品实例的方法getHouse，在这个方法中初始化实例
#### c++实现
```c++

```
#### go实现
```go
package builder

import "fmt"

// 产品
type House struct {
	windows string
	door    string
	floor   string
}

// 生成器接口
type IBuilder interface {
	setWindows()
	setDoor()
	setFloor()
	// 返回对象
	getHouse() House
}

// 具体nomal生成器
type NormalBuilder struct {
	// 生成器和有实例的元素
	windows string
	door    string
	floor   string
}

func (n *NormalBuilder) setWindows() {
	n.windows = "normal windows"
}

func (n *NormalBuilder) setDoor() {
	n.door = "normal door"
}

func (n *NormalBuilder) setFloor() {
	n.floor = "normal floor"
}

func (n *NormalBuilder) getHouse() House {
	return House{
		windows: n.windows,
		door:    n.door,
		floor:   n.floor,
	}
}

func newNormalBuilder() *NormalBuilder {
	return &NormalBuilder{}
}

// 具体冰屋生成器
type IceBuilder struct {
	windows string
	door    string
	floor   string
}

func (i *IceBuilder) setWindows() {
	i.windows = "ice windows"
}

func (i *IceBuilder) setDoor() {
	i.door = "ice door"
}

func (i *IceBuilder) setFloor() {
	i.floor = "ice floor"
}

func (i *IceBuilder) getHouse() House {
	return House{
		floor:   i.floor,
		door:    i.door,
		windows: i.windows,
	}
}

func newIceBuilder() *IceBuilder {
	return &IceBuilder{}
}

// 主管：生产一个产品
type Director struct {
	builder IBuilder
}

func (d *Director) buildHouse() House { // 指定顺序
	d.builder.setDoor()
	d.builder.setFloor()
	// d.builder.setWindows()
	return d.builder.getHouse()
}

// 创建一个主管实例
func newDirector(b IBuilder) *Director {
	return &Director{
		builder: b,
	}
}

func Builder() {
	// 创建一个builder
	nb := newNormalBuilder()
	// 创建一个主管实例
	nd := newDirector(nb)
	nh := nd.buildHouse()
	fmt.Println(nh.door)
	fmt.Println(nh.floor)
	fmt.Println(nh.windows)
}
```
### 扩展
- 扩展一个火屋
	- 实现生成器接口
	- 返回自己（只要是方便客户端调用）
```go
type FireBuilder struct {
	floor   string
	windows string
	door    string
}

func (f *FireBuilder) setDoor() {
	f.door = "fire door"
}

func (f *FireBuilder) setFloor() {
	f.floor = "fire floor"
}

func (f *FireBuilder) setWindows() {
	f.windows = "fire windows"
}

func (f *FireBuilder) getHouse() House {
	return House{
		floor:   f.floor,
		door:    f.door,
		windows: f.windows,
	}
}

func newFireBuiler() *FireBuilder {
	return &FireBuilder{}
}
```