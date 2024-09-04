### go调试源代码
#### 修改go源码
- 打开源码
```go
func Println(a ...any) (n int, err error) {
	return Fprintln(os.Stdout, a...)
}
	```
- 添加运行时函数之后编译
```go
func Println(a ...any) (n int, err error) {
	println("baike")
	return Fprintln(os.Stdout, a...)
}
```
- 使用脚本编译
```shell
D:\go\src\make.bash 
```
#### 中间代码生成
- go中间代码有静态单赋值特性
```go
go build -gcflags -S main.go
```
- 还可以生成一个ssa.html文件，查看汇编优化的每一个步骤
```go
GOSSAFUNC=main go build main.go
```

### 编译过程
