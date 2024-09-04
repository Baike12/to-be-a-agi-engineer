### 接口是什么
- 接口一组方法的签名
#### 什么时候检查
- 传递参数、返回参数、变量赋值
- 只在需要的时候检查
#### 类型
- 接口也是一种类型
##### 带方法的接口
##### 不带方法的接口

#### 接口和指针
##### 使用结构体和结构体指针实现接口
- 使用结构体初始化的变量不能调用使用结构体指针实现的接口
- 使用指针初始化的变量可以调用结构体实现的接口，是因为指针可以隐式获取结构体
- 原因
	- go参数传递都是传值的
	- 结构体指针是传递指针，还是指向原来的结构体
	- 结构体就会拷贝一份，没有指针，就算有指针，也不好控制指向原来的结构体
#### 接口的隐式类型转换
- 在传参时和变量赋值时发生
- 其他类型ot转换成接口类型it还会包含ot的类型信息，所以不是nil

### 接口类型结构
#### 不带方法
```go
type eface struct {
	_type *_type
	data  unsafe.Pointer
}
```
其中类型
```go
type Type struct {
	Size_       uintptr
	PtrBytes    uintptr // number of (prefix) bytes in the type that can contain pointers
	Hash        uint32  // hash of type; avoids computation in hash tables
	TFlag       TFlag   // extra type information flags
	Align_      uint8   // alignment of variable with this type
	FieldAlign_ uint8   // alignment of struct field with this type
	Kind_       uint8   // enumeration for C
	// function for comparing objects of this type
	// (ptr to object A, ptr to object B) -> ==?
	Equal func(unsafe.Pointer, unsafe.Pointer) bool
	// GCData stores the GC type data for the garbage collector.
	// If the KindGCProg bit is set in kind, GCData is a GC program.
	// Otherwise it is a ptrmask bitmap. See mbitmap.go for details.
	GCData    *byte
	Str       NameOff // string form
	PtrToThis TypeOff // type for pointer to this type, may be zero
}
```
- size是大小
- hash用来快速判断两个类型是否相等
#### 带方法的接口
```go
type iface struct {
	tab  *itab
	data unsafe.Pointer
}
```
其中itab
```go
type itab struct {
	inter *interfacetype
	_type *_type
	hash  uint32 // copy of _type.hash. Used for type switches.
	_     [4]byte
	fun   [1]uintptr // variable sized. fun[0]==0 means _type does not implement inter.
}
```
- hash是从type中拷贝来的，用于类型断言
- func是虚函数表，存储函数指针，用于动态派发
### 类型转换
#### 指针类型
在编译期间将一些需要动态派发的方法调用改成对目标方法的直接调用
#### 结构体类型
会在栈上初始化结构体，结构体中只有一个string
- 初始化一个stringheader：data指向字面量，长度计算出来
调用convt2i
- 这个函数获取itab中的类型，并根据类型申请一段空间
- convt2i返回一个iface，其中包含itab和cat变量
- 再通过结构体调用接口方法时，会从这个结构体的itab中的func数组中获取方法

### 类型断言
- 将接口类型转换成具体类型
#### 非空接口
- 将接口的itab.hash的值与要比较的目标类型的hash值进行比对
#### 空接口
- 也是类似用接口type中的hash值与目标类型hash值进行比对

### 动态派发
- 调用接口类型的方法时，当编译时不能确定接口的类型，会在运行时决定调用方法的哪个实现
#### 动态派发流程
- 从接口的itab中获取方法f
- 将接口的data拷贝到栈
- f会被拷贝到寄存器然后使用call触发
#### 动态派发会带来性能损耗
- 结构体指针编译器优化之后会降低到百分之五左右
- 结构体的动态派发损耗在一倍左右，应该避免使用，主要是因为传值带来的拷贝