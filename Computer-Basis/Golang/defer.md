### defer
#### 作用域
- 在函数返回之前调用
#### 预计算参数
- defer在调用时就拷贝参数了
- 对defer传入函数指针可以使函数在返回之前再执行而不是在调用defer的时候就执行
```go
func myDefer() {
	start := time.Now()
	defer func() { fmt.Println(time.Since(start)) }()
	time.Sleep(time.Second * 1)
}
```
#### 数据结构
```go
type _defer struct {
	heap      bool
	rangefunc bool    // true for rangefunc list
	sp        uintptr // sp at time of defer
	pc        uintptr // pc at time of defer
	fn        func()  // can be nil for open-coded defers
	link      *_defer // next defer on G; can point to either heap or stack!

	// If rangefunc is true, *head is the head of the atomic linked list
	// during a range-over-func execution.
	head *atomic.Pointer[_defer]
}
```
#### 执行机制
可以使用开放编码、堆分配和栈分配来处理defer，开放编码使defer的开销变小