### bridge模式是什么
- 将一个大类或者一系列紧密相关的类拆分成抽象和实现两个独立层次
### 适用场景
- 场景：
	- 有一个形状类，能扩展出圆形和方形两个子类，希望添加颜色，成为红色圆形和蓝色方形等
	- 问题在于之后再添加一个三角形，就需要添加红色三角形和蓝色三角形，之后再添加绿色就更糟糕
- 问题：希望在两个维度上扩展类
- 解决方法：将继承改为组合，这样一个类就不会有颜色和形状两个维度
- 用于
	- 拆分或重组一个具有多重功能的庞杂类
	- 在多个独立维度上扩展类
	- 在运行时切换不同实现方法
### 实现
- 假设有mac和windows计算机、爱普生和惠普打印机，他们会以任意方式组合起来执行打印动作
- 抽象层：计算机
- 实现层：打印机
#### c++实现
```c++

```
#### go实现
```go
package bridge

import "fmt"

// 抽象
type Computer interface {
	Print()
	SetPrinter(Printer)
}

// 实现
type Printer interface {
	PrintFile()
}

// mac电脑实现电脑接口，作为具体的抽象
type Mac struct {
	printer Printer
}

func (m *Mac) Print() {
	fmt.Println("Mac  print")
	m.printer.PrintFile()
}

// 使用打印机之前要先设置打印机
func (m *Mac) SetPrinter(p Printer) { // 设置mac的打印机
	m.printer = p
}

// windows实现电脑接口，作为另一个具体的抽象
type Windows struct {
	printer Printer
}

func (w *Windows) Print() {
	fmt.Println("Windows print")
	w.printer.PrintFile()
}

func (w *Windows) SetPrinter(p Printer) {
	w.printer = p
}

// 具体的打印机实现打印动作
type Epson struct {
}

func (e *Epson) PrintFile() {
	fmt.Println("print by epson")
}

type Hp struct {
}

func (h *Hp) PrintFile() {
	fmt.Println("print by hp")
}

func Bridge() {
	// 初始化打印机
	ep := &Epson{}
	// 初始化电脑
	mac := &Mac{}
	// 设置打印机
	mac.SetPrinter(ep)
	mac.Print() // mac打印// 实际是mac中的打印机打印的，mac中的打印机之前已经被设置了
}

```
### 扩展
- 添加打印机Lenovo，只需要实现打印机接口，然后set的时候设置Lenovo打印机就行