## 反射
两个函数
- reflect.TypeOf
- reflect.ValueOf
两个类型
- reflect.Type
- reflect.Value
### 三大法则
#### 从interface变量可以反射出反射对象
得到反射对象的值和类型之后就可以通过Type接口定义的函数和类型字段等，类型是结构体、函数、map等也可以通过相应的方法得到类型的定义
```go
func myreflect() {
	s := "nihao"
	fmt.Println(reflect.ValueOf(s))
	fmt.Println(reflect.TypeOf(s))
}
```
- 这里因为ValueOf的参数是接口类型，所以实际上发生了类型转换
#### 可以从反射对象获取interface变量
不管是反射对象到接口值还是接口值到反射对象，都要先经过接口类型转换
```go
func myreflect() {
	s := "nihao"
	fmt.Println(reflect.ValueOf(s))
	fmt.Println(reflect.TypeOf(s))
	v := reflect.ValueOf(s) // 反射对象
	fmt.Println(reflect.TypeOf(v))
	fmt.Println(v.Interface()) // 获得原始值1// 将反射对象转换成原始值
}
```
#### 如果要修改反射变量，它的值必须是可以设置的
```go
	// v.SetString("wohao")// 因为传参值拷贝，所以v和s没什么关系了已经，为了防止错误程序会崩溃
	// fmt.Println(s)
	v1 := reflect.ValueOf(&s)
	v1.Elem().SetString("wohao")
	fmt.Println(s)
```
### 类型和值
#### TypeOf实现原理
将interface转换成reflect.emptyInterface，然后获取相应的类型信息
#### Value实现原理
- 先使用escape保证当前值逃逸到堆上
- 然后通过unpackeface从接口中获取value结构体，这个unpackeface会将传入的接口转换成emptyinterface，然后将具体类型和指针包装成Value结构后返回
在讲一个变量转换成反射对象时，编译期间就把变量转换成interface类型了，等待运行期间获取reflect包获取信息
### 更新变量
### 实现协议
#### 判断一个类型是否遵循特定接口
- 获取接口的类型：将nil转换成接口指针，再获取指针指向的接口，对接口进行typeOf
