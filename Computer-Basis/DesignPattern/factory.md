### factory是什么
- 抽象工厂：一个接口，包含一个抽象的工厂方法
- 具体工厂：实现抽象工厂接口，创建具体产品
- 抽象产品：一个接口，定义产品
- 具体产品：实现抽象产品接口
### 适用场景
- 创建对象涉及复杂初始化，并且每种对象的初始化不一样
### 实现
- 
#### c++实现
```c++

```
#### go实现
```go
package factory

import "fmt"

type Block interface { // 抽象产品
	produce() // 创建产品
}

type CircleBlock struct{} // 具体产品

func (cb *CircleBlock) produce() {
	fmt.Println("circle block")
}

type SquareBlock struct{}

func (sb *SquareBlock) produce() {
	fmt.Println("square block")
}

type BlockFactory interface { // 抽象工厂
	createBlock() Block
}

type CirlcleFactory struct{} // 具体工厂

func (cf *CirlcleFactory) createBlock() Block { // 具体工厂创建具体产品
	return &CircleBlock{}
}

type SquareFactory struct{}

func (sf *SquareFactory) createBlock() Block {
	return &SquareBlock{}
}
```
### 扩展
- 添加具体产品，实现抽象产品接口
- 添加具体工厂，实现抽象工厂接口