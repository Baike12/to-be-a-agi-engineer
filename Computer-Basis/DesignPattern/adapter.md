### adapter是什么
- 一个对象，能够转换对象接口，使其能与其他对象交互
### 适用场景
- 一个类的接口与其他类不适配
### 实现
- 
#### c++实现
```c++

```
#### go实现
```go
package adapter

import "fmt"

type Computer interface {
	InsertIntoLinghtningPort()
}

type Client struct {
}

func (c *Client) InsertLightingIntoComputer(com Computer) { // 客户端的目的是想把l型接口插入电脑中
	fmt.Println("client insert lighting into computer")
	com.InsertIntoLinghtningPort()
}

type Mac struct {
}

func (m *Mac) InsertIntoLinghtningPort() { // 实现了电脑接口
	fmt.Println("lightning port inserted into Mac")
}

type Windows struct {
}

func (w *Windows) InsertIntoUSBPort() {
	fmt.Println("usb port inserted into windows")
}

// 问题来了：windows不支持被lightning插入（没实现computer接口）
type WindowsAdapter struct {
	WindowsMachine Windows // 将windows的类型封装在适配器中
}

// 适配器来实现电脑接口
func (w *WindowsAdapter) InsertIntoLinghtningPort() {
	fmt.Println("adapter convert lightning port to usb port")
	w.WindowsMachine.InsertIntoUSBPort() // 在适配器里面再来进程转换
}

func Adapter() {
	client := &Client{}
	mac := &Mac{}

	client.InsertLightingIntoComputer(mac) // 客户端将lightning插入mac，这是本来就支持的

	win := &Windows{}
	winAda := &WindowsAdapter{
		WindowsMachine: *win,
	}

	client.InsertLightingIntoComputer(winAda)
}

```
### 扩展
- 新增适配器 
```go
// 扩展linux接口
type Linux struct {
}

func (l *Linux) InsertIntoLinuxPort() {
	fmt.Println("linux port inserted to linux")
}

// 新增是配置，适配器来实现computer接口
type LinuxAdapter struct {
	LinuxMachine Linux
}

func (la *LinuxAdapter) InsertIntoLinghtningPort() {
	fmt.Println("linux adapter convert lighting port to linux port")
	la.LinuxMachine.InsertIntoLinuxPort() // 转换成linux的接口
}
```