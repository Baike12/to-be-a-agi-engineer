### decorator模式是什么
- 可以将一个对象放到包含行为的特殊封装对象中实现为原来的对象绑定新的行为
- 聚合：一个对象中a包含对另一个对象b的引用，在b中实现a希望的额外动作
### 适用场景
- 无需修改对象代码就可以为对象添加额外行为
- 对于有final限制的类，只能使用装饰器来封装
- 可以在运行时扩展类的动作
### 实现
- 定义一个组件接口，后面的具体组件和装饰器都实现这个接口
- 定义具体组件，实现接口
- 定义装饰器，包含组件接口，并且自身也实现组件接口，这样装饰器能包含具体组件，还能作为组件（因为它实现了组件接口）
#### c++实现
```c++

```
#### go实现
```go
package decorator

import "fmt"

// 定义食物组件接口
type Food interface {
	getPrice() int
}

// 定义煎饼组件
type Pie struct {
	price int
}

func (p *Pie) getPrice() int {
	return p.price
}

// 定义鸡蛋装饰器
type Egg struct {
	food Food
}

func (e *Egg) getPrice() int {
	price := e.food.getPrice()
	return price + 2
}

// 火腿肠装饰器
type Sausage struct {
	food Food
}

func (s *Sausage) getPrice() int {
	price := s.food.getPrice()
	return price + 3
}

func Decorator() {
	pie := &Pie{price: 6}

	pieWithEgg := &Egg{
		food: pie,
	}

	pieWithEggAndSausage := &Sausage{
		food: pieWithEgg,
	}

	price := pieWithEggAndSausage.getPrice()
	fmt.Println(price)
}

```
### 扩展
- 