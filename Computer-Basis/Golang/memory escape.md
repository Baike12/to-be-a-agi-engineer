# 1	面试
## 1.1	什么是内存逃逸
函数内部创建的变量或对象，在函数结束后仍然被其他部分持有或引用
## 1.2	为什么问
### 1.2.1	内存管理
- 栈上内存由编译器管理，速度快但是空间有限
- 堆上内存分配要运行时参与，相对比较慢但是空间大，由gc回收
**内存分配原则：**
- 指向栈上对象的指针不能存放在堆中，否则会将对象移动到堆
- 栈上指针的生命周期不能超过栈上对象的生命周期，否则也移动到堆上
### 1.2.2	逃逸影响
- 分配和回收
- 内存安全：主要是指针引用
- 内存泄露
### 1.2.3	逃逸场景
并不是所有逃逸都是坏的，闭包就是利用内存逃逸
## 1.3	原因
### 1.3.1	栈空间与作用域
### 1.3.2	编译时无法确定大小
- interface动态确定
- 切片扩展
## 1.4	逃逸情况分析
使用：
```shell
go build -gcflags "all=-N -l -m" -o a main.go
```
### 1.4.1	指针、切片和map作为返回值
编译器无法确定这些变量何时停止使用，所以只能放到堆上
### 1.4.2	向channel中发送包含指针或者指向数据的指针
编译器无法确定消息什么时候被接收
### 1.4.3	闭包
闭包函数使用了外层函数的变量
### 1.4.4	slice或者map中存储包含指针或者指针
### 1.4.5	未指定具体类型的接口
### 1.4.6	超过了栈内存
```go
package memoryescape

import "fmt"

func pointerEscape() (*int, []int, map[int]int) {
	i := 10
	l := []int{1, 2, 3}
	m := map[int]int{1: 1, 2: 2}
	return &i, l, m
}

func ptrInChan() {
	i := 2
	ch := make(chan *int, 2)
	ch <- &i
	<-ch
}

func exceedStackEscape() {
	s := make([]int, 0, 10000)
	for idx, _ := range s {
		s[idx] = idx
	}
}

func noCertainSizeEscape(size int) {
	s := make([]int, size)
	fmt.Println(s)
}

type animal interface {
	run()
}

type dog struct{}

func (d *dog) run() {}

func dynamicTypeEscape() {
	var a animal
	a = &dog{}
	a.run()
}

func closureEscape() func() int {
	i := 1
	return func() int {
		i++
		return i
	}
}
```
![[Pasted image 20240922222829.png]]