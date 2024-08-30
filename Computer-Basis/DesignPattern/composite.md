### composite模式是什么
- 将对象组合成树状结构
### 适用场景
- 有一个盒子，盒子中可能有小盒子和产品，计算产品总价
	- 打开所有盒子计算产品总价：代码中不好实现
- 解决方法：使用一个通用接口来与盒子和产品交互，在接口中声明一个计算总价的方法
	- 如果是盒子就打开盒子，是产品就计算总价
- 适用：
	- 树状对象结构
	- 用相同的方式处理复杂和简单元素
### 实现
- 在文件系统中搜索关键字
#### c++实现
```c++

```
#### go实现
```go
package composite

import "fmt"

// 组件接口，组件包含文件叶子结点和文件夹，文件夹实际上是一个包含文件夹和文件的组合
type Component interface {
	search(string)
}

type Folder struct {
	components []Component // 文件夹中包含一系列文件和文件夹
	name       string      // 文件夹名字
}

// 文件夹实现组件接口
func (f *Folder) search(keyword string) {
	// 文件夹遍历自己的子节点
	fmt.Printf("search keyword %s from %s\n", keyword, f.name)
	for _, composite := range f.components {
		composite.search(keyword)
	}
}

// 文件夹提供一个接口添加composites
func (f *Folder) add(c Component) {
	f.components = append(f.components, c)
}

// 文件实现component接口
type File struct {
	name string
}

func (f *File) search(keyword string) {
	fmt.Printf("search keyword %s from %s\n", keyword, f.name)
}

func (f *File) getName() string {
	return f.name
}

func Composite() {
	// 创建几个文件
	f1 := &File{name: "f1"}
	f2 := &File{name: "f2"}
	f3 := &File{name: "f3"}

	fo1 := &Folder{name: "fo1"}

	fo1.add(f1)
	fo1.search("rose")
	fo2 := &Folder{name: "fo2"}

	fo2.add(f2)
	fo2.add(f3)
	fo2.add(fo1)
	fo2.search("mike")
}

```
### 扩展
- 添加要