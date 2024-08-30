### abstract factory是什么
- 工厂模式一个工厂只能创建一个产品，抽象工厂一个工厂可以创建一系列产品
### 适用场景
- 创建一系列相关或者有依赖关系的对象，比如与不同数据库的连接对象
- 抽象工厂提供一个接口，用于创建每个系列产品的对象，只要代码通过该接口创建对象，就不会生成与其他已经生成的对象类型不一致的对象，比如跨平台UI
### 实现
- 抽象工厂接口：
	- 定义抽象产品创建方法，这些方法返回抽象产品接口
	- 客户端通过指定产品类型获得具体工厂
- 具体工厂
	- 实现抽象工厂的方法，返回具体产品
- 抽象产品接口
	- 定义产品行为并基于抽象产品实现产品行为
- 具体产品
	- 包装抽象产品，因为抽象产品已经实现了产品行为，所以具体产品也就实现了产品行为
- 总结：
	- 工厂是一个维度，选择了一个工厂就能生产这个工厂的产品，这些产品都来自同一个工厂
#### c++实现
```c++

```
#### go实现
```go
package abstractfactory

import "fmt"

// 抽象产品接口1
type IShoe interface {
	setLogo(logo string)
	getLogo() string
}

// 抽象产品
type Shoe struct {
	logo string
}

func (s *Shoe) setLogo(logo string) {
	s.logo = logo
}

func (s *Shoe) getLogo() string {
	return s.logo
}

// 具体产品11
type AdidasShoe struct {
	Shoe
}

// 具体产品12
type NikeShoe struct {
	Shoe
}

// 抽象产品接口2
type IShirt interface {
	setLogo(logo string)
	getLogo() string
}

type Shirt struct {
	logo string
}

func (s *Shirt) setLogo(logo string) {
	s.logo = logo
}

func (s *Shirt) getLogo() string {
	return s.logo
}

// 具体产品21
type AdidasShirt struct {
	Shirt
}

// 具体产品22
type NikeShirt struct {
	Shirt
}

// 抽象工厂接口：其中为抽象产品定义构建方法
type SportFactory interface {
	makeShoe() IShoe
	makeShirt() IShirt
}

// 具体工厂1
type Adidas struct{}

func (a *Adidas) makeShoe() IShoe {
	fmt.Println("adidas shoe")
	return &AdidasShoe{
		Shoe: Shoe{
			logo: "adidas log",
		},
	}
}

func (a *Adidas) makeShirt() IShirt {
	fmt.Println("adidas shirt")
	return &AdidasShirt{
		Shirt: Shirt{
			logo: "adidas log",
		},
	}
}

// 具体工厂2
type Nike struct{}

func (n *Nike) makeShoe() IShoe {
	fmt.Println("nike shoe")
	return &NikeShoe{
		Shoe: Shoe{
			logo: "nike logo",
		},
	}
}

func (n *Nike) makeShirt() IShirt {
	fmt.Println("nike shirt")
	return &NikeShirt{
		Shirt: Shirt{
			logo: "nike logo",
		},
	}
}

// 根据类型获取工厂方法
func GetFactory(brand string) (SportFactory, error) {
	if brand == "adidas" {
		return &Adidas{}, nil
	} else if brand == "nike" {
		return &Nike{}, nil
	} else {
		fmt.Println("there is no factory")
		return nil, nil
	}
}

func AbstractFactory() {
	f1, err := GetFactory("adidas")
	if err != nil {
		return
	}
	f1.makeShoe()
	f1.makeShirt()

	f2, err := GetFactory("nike")
	if err != nil {
		return
	}
	f2.makeShoe()
	f2.makeShirt()
}

```
### 扩展
- 添加具体工厂
- 添加具体产品，具体产品直接包含抽象产品就行，因为抽象产品已经实现了抽象产品接口
- 修改获取抽象工厂的方法
```go
// 具体产品
type AntaShoe struct {
	Shoe
}

type AntaShirt struct {
	Shirt
}

// 添加具体工厂
type Anta struct{}

func (an *Anta) makeShoe() IShoe {
	fmt.Println("anta shoe")
	return &AdidasShoe{
		Shoe: Shoe{
			logo: "anta logo",
		},
	}
}

func (an *Anta) makeShirt() IShirt {
	fmt.Println("anta shirt")
	return &AntaShirt{
		Shirt: Shirt{
			logo: "logo",
		},
	}
}

// 根据类型获取工厂方法
func GetFactory(brand string) (SportFactory, error) {
	if brand == "adidas" {
		return &Adidas{}, nil
	} else if brand == "nike" {
		return &Nike{}, nil
	} else if brand == "anta" {
		return &Anta{}, nil
	} else {
		fmt.Println("there is no factory")
		return nil, nil
	}
}
```