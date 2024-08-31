### 题号：27 removeElement
#### 思路：
- 双指针：快指针遇到与val不同的就覆盖慢指针，最后返回慢指针的值
#### c++实现：
```c++

```
#### go实现：
```go
func removeElement(nums []int, val int) int {
	slow, fast := 0, 0

	for fast < len(nums) {
		if nums[fast] != val {
			nums[slow] = nums[fast]
			slow++
		}
		fast++
	}
	return slow
}
```
#### 总结：
- 注意一下下标就行