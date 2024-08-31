### 题号：344 reverse
#### 思路：
- 
#### c++实现：
```c++

```
#### go实现：
```go
func reverseString(s []byte) {
	l := 0
	r := len(s)-1
	for l < r{
		s[l], s[r] = s[r], s[l]
		l++
		r--
	}
}
```
### 题号：541 reverseStr
#### 思路：
- 从当前下标开始，判断后面是否还有k个，有的话对前k个翻转，没有的话就对剩下的翻转
#### c++实现：
```c++

```
#### go实现：
```go
func reverseStr(s string, k int) string {
	bs := []byte(s)
	for pos := 0; pos < len(s); pos += 2 * k {
		if pos+k < len(s) {
			l := pos
			r := pos + k - 1
			for l < r {
				bs[l], bs[r] = bs[r], bs[l]
				l++
				r--
			}
		} else {
			l := pos
			r := len(s) - 1
			for l < r {
				bs[l], bs[r] = bs[r], bs[l]
				l++
				r--
			}
		}
	}
	return string(bs)
}
```
### 题号：replace number
#### 思路：
- 这个题用go实现没有什么特别的技巧，因为go的字符串不能修改，
- 如果用c++
	- 统计数字个数，将字符串扩容到替换后的长度
	- 双指针
		- p1指向原字符串尾，p2指向扩容后的字符串尾
		- p1前向扫描，是字符直接移动过去，是数组转换成number添加到后面
#### c++实现：
```c++

```
#### go实现：
```go
func replaceNumber() {
	var strByte []byte

	fmt.Scanln(&strByte)

	for i := 0; i < len(strByte); i++ {
		if strByte[i] <= '9' && strByte[i] >= '0' {
			insertByte := []byte{'n', 'u', 'm', 'b', 'e', 'r'}
			strByte = append(strByte[:i], append(insertByte, strByte[i+1:]...)...) // 加上...表示切片中的元素作为单独的元素进行append
			i = i + len(insertByte) - 1
		}
	}
	fmt.Printf(string(strByte))
}
```
### 题号：151 reverseWord
#### 思路：
- 先把多余空格去掉
- 再把整个字符串翻转过来
- 最后把每一个单词反过来
#### c++实现：
```c++

```
#### go实现：
```go
func reverseWords(s string) string {
	bs := []byte(s)

	reverse(bs)
	fmt.Println(string(bs))
	l, r := 0, 0
	// 删除头部空格
	for len(bs) > 0 && r < len(bs) && bs[r] == ' ' {
		r++
	}

	for ; r < len(bs); r++ {
		// 连续空格直接右移不拷贝，当前下标和当前下标的左侧都是空格，前提是当前下标有左侧，因为头部空格已经被跳过了，所以 > 就行
		if r-1 > 0 && bs[r] == bs[r-1] && bs[r] == ' ' {
			continue
		}
		bs[l] = bs[r]
		l++
	}

	// 删除尾部空格：// 经过上面删除中间空格之后尾部只会有一个空格
	if bs[l-1] == ' ' {
		bs = bs[:l-1]
	} else {
		bs = bs[:l]
	}
	// 翻转每一个单词
	i := 0
	for i < len(bs) {
		j := i
		for ; j < len(bs) && bs[j] != ' '; j++ {
		}
		reverse(bs[i:j])
		i = j + 1
	}
	return string(bs)
}
```
#### 总结：
- 下标操作要清晰
### 题号：55 right roatte
#### 思路：
- 先整体倒序
- 再局部倒序
#### c++实现：
```c++

```
#### go实现：
```go
func rotateStr(s string, k int) string {
	bs := []byte(s)

	reverse(bs)
	reverse(bs[:k])
	reverse(bs[k:])
	return string(bs)
}
```
#### 总结
- 对于移动序列的题目，旋转是一个可以考虑的点
### 题号：28 strStr
#### 思路：
- 前缀表：
	- 是什么：用于回退，记录模式串与主串不匹配时，模式串应该从哪开始重新匹配
	- 有什么用：记录下标i（包括）之前的字符串中，有多大长度的相同前后缀
		- 前缀：不包括最后一个字符的所有以第一个字符开头的连续子串
		- 后缀：不包括第一个字符的所有以最后一个字符结尾的连续子串
	- 计算前缀表（前缀表值减1的情况）例子：aabaaf：-1,0,-1,0,1,-1
		- 定义两个指针：j表示最长相等前后缀长度，i指向后缀末尾
		- 初始化j为-1，本来是0，但是因为所有值都减一所以是-1，对前缀表的第一个元素赋值为j，也就是-1，表示只有第一个元素的序列最大公共前后缀数量为-1
			- 从1开始递增i
			- 如果模式串中的当前前后缀不匹配且j不小于0时，将j回退到前一位，这实际上是在找一个更短的相同前后缀，并给j赋值，j就是最长相同前后缀
			- 如果当前
- kmp流程
	- 如果模式串长度为0，就返回0
	- 创建next数组
	- 获取next数组
	- 初始化模式串起始位置为-1
	- 从0开始遍历主串
		- 如果主串的i位置不等于模式串的j+1位置（为什么要 + 1，因为j被初始化为-1了）
			- 那么要寻找主串和模式串匹配的下一个位置
			- 注意这里要用循环，因为不一定一次寻找就找到了
		- 当然，要是主串的i位置 = 模式串的j + 1位置，那模式串 + 1，外层循环中会将i + 1的
		- 如果j指向了模式串尾，那么就返回当前位置，注意这里j = = 模式串长度 - 1就行，不用 - 2，因为先进行了上面的流程在判断的，j已经被 + 1一次了
	- 如果知道最后模式串都没有指到最后，返回-1
#### c++实现：
```c++

```
#### go实现：
```go
func getNext(next []int, s string) {
	j := -1
	next[0] = j // 表明j就是最长相同前后缀

	for i := 1; i < len(s); i++ {
		for j >= 0 && s[i] != s[j+1] { // i是当前正在考虑的字符，j+1是最长相同前后缀的下一个字符
			j = next[j] // 找到上一个最长相同前后缀的位置
		} // 直到有一个前缀的下一个可以和i相同

		if s[i] == s[j+1] {
			j++
		}
		// 将j赋值给next数组
		next[i] = j
	}
}

func strStr(haystack string, needle string) int {
	if len(needle) == 0{
		return 0
	}

	next:= make([]int, len(needle))
	getNext(next, needle)

	j := -1
	for i :=0;i<len(haystack);i++{
		for j >=0 && haystack[i]!= needle[j+1]{
			j = next[j]
		}
		if haystack[i]==needle[j+1]{
			j++
		}
		if j == len(needle)-1{
			return i - len(needle)+1
		}
	}
	return -1
}
```
#### 总结：
- 代码看不懂别硬看，先一句一句翻译成自然语言，然后看看能不能理解，还是不理解再使用调试工具跟一遍，一般跟一遍就理解了，要是还是不理解，就把代码背下来，也许啥时候灵光一闪就理解了
### 题号：459 repeatedSubstringPattern
#### 思路：
- 能组成s的是否必须是开头几个字符？是的
- 方法一：组合两个串
	- 这里有一个结论：如果s可以由多个子串拼起来，那么两个s拼起来一定可以再找到一个s，原因如下
		- 如果s可以由偶数个subs拼起来，那么是明显的
		- 奇数个，看起来也是明显的
#### c++实现：
```c++

```
#### go实现：
```go
func repeatedSubstringPattern(s string) bool {
	if len(s) == 0 {
		return false
	}

	ss := s + s
	return strings.Contains(ss[1:2*len(s)-1], s)
}
```
#### 总结：
- 有些题可以总结一些规律
	- 对于字符串可以试试字符串本身旋转和字符串本身拼接