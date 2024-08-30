### chain of command模式是什么
- 将请求沿着处理者链发送，每个处理者可以对请求进行处理，或者将请求传递给链上的下一个处理者
### 适用场景
- 需要使用不同方式处理不同种类的请求，而且请求类型和顺序事先未知时适用责任链模式
- 所需处理者和顺序需要在运行时改变时
### 实现
- 所有处理者都实现相同接口
- 具体实现
	- 一个患者依次到：前台、医生、药房、收银处
#### c++实现
```c++

```
#### go实现
```go
package cor

import "fmt"

// 定义患者
type Patient struct {
	name         string
	registerDone bool
	doctorDone   bool
	medicineDone bool
	payDone      bool
}

// 定义部门接口，后序每个部门都要实现这个接口
type Department interface {
	execute(*Patient) // 个部分要执行的操作，执行的时候还要根据情况指定下一个部门
	setNext(Department)
}

// 前台
type Register struct {
	next Department
}

func (r *Register) execute(p *Patient) {
	if p.registerDone {
		fmt.Println("patient had registered")
		r.next.execute(p)
		return
	}
	fmt.Println("patient registering")
	p.registerDone = true
	r.next.execute(p)
}

func (r *Register) setNext(next Department) {
	r.next = next
}

type Doctor struct {
	next Department
}

func (d *Doctor) execute(p *Patient) {
	if p.registerDone {
		fmt.Println("patient had doctored")
		// d.next.execute(p)
		return
	}
	fmt.Println("patient doctoring")
	p.doctorDone = true
	// d.next.execute(p)
}

func (d *Doctor) setNext(next Department) {
	d.next = next
}

func Cor() {
	// 初始化医生
	doctor := &Doctor{}
	// 初始化前台，并把前台的next设置为医生
	register := &Register{}
	register.setNext(doctor)
	// 初始化患者，并从前台开始调用责任链
	p := &Patient{name: "zhangsan"}
	register.execute(p)
}

```
### 扩展
- 添加一个处理者，并且设置这个处理者的上下游