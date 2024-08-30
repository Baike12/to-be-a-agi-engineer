### command模式是什么
- 将请求转换成一个包含与请求相关的所有信息的对象，这种转换能根据不同请求将方法参数化、延迟请求执行或者将其放到队列中，并且可以实现可撤销操作
- 发送请求的对象只需知道如何发送请求，不用知道请求如何完成
### 适用场景
- 需要通过操作来参数化对象
- 需要将操作放到队列中
- 需要实现操作回滚
### 实现
- 调用者：触发命令，但是不直接发送请求，不创建命令对象，通常通过构造函数从客户端获得预先生成的命令
- 命令：只声明一个执行命令的方法
- 具体命令：实现各种类型请求，将命令委派给业务逻辑对象
- 接受者：处理业务
- 客户端：创建并配置命令对象，客户端必须将所有请求参数传递给命令构造函数
#### c++实现
```c++

```
#### go实现
```go
package command

import "fmt"

// 定义请求者// 请求者包含一条命令
type Button struct {
	command Command
}

// 请求者执行一个动作，这个动作通过命令来执行
func (b *Button) press() {
	b.command.execute()
}

// 定义命令接口
type Command interface {
	execute()
}

// 具体命令
type OnCommand struct {
	device Device // 具体命令包含接受者
}

// 具体命令实现命令接口
func (on *OnCommand) execute() {
	on.device.on() // 在具体命令中执行动作
}

// 另一个具体命令，还是一样包含接受者，实现命令接口
type OffCommand struct {
	device Device
}

func (of *OffCommand) execute() {
	of.device.off()
}

// 定义接受者接口
type Device interface {
	on()
	off()
}

// 具体接受者
type Tv struct {
	isRunning bool
}

func (t *Tv) on() {
	t.isRunning = true
	fmt.Println("turning tv on")
}

func (t *Tv) off() {
	t.isRunning = false
	fmt.Println("turning tv off")
}

func CommandTyle() {
	// 创建接受者
	tv := &Tv{}
	// 使用接受者创建命令
	onCommand := &OnCommand{device: tv}
	// 使用命令创建请求者
	onButton := &Button{command: onCommand}
	onButton.press()
}

```
### 扩展
- 请求者、命令和接受者都可以