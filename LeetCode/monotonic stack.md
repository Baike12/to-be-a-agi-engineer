### 题号：739 dailyTemperatures
#### 思路：
- 单调栈：适用于一维数组，要寻找一个元素的右边或者左边第一个比自己大或者小的元素的位置
	- 单调栈递增（栈顶到栈底，也就是越深越大）：求右边第一个比自己大的
	- 单调栈递减（栈顶到栈底，越深越小）：求左边第一个比自己大的
	- 栈中放的是元素下标
- 流程
	- 下标便利数组
	- 当前遍历的元素的值 <  = 栈顶元素对应的值（这个值直接根据下标从原数组取），直接压栈
	- 当前遍历的元素的值 > 栈顶元素对应的值
		- 将栈中元素弹出
		- 结果中栈顶元素的值 = 当前下标 - 栈顶元素
#### c++实现：
```c++

```
#### go实现：
```go
func dailyTemperatures(temperatures []int) []int {
	res := make([]int, len(temperatures))

	stack := []int{0} // 栈顶元素初始化为0，这是下标

	for i := 1; i < len(temperatures); i++ {
		top := stack[len(stack)-1] // 栈顶元素

		if temperatures[i] <= temperatures[top] {
			stack = append(stack, i)
		} else {
			for len(stack) != 0 && temperatures[i] > temperatures[top] {
				res[top] = i - top // 现在的i就是第一个比top位置大的
				stack = stack[:len(stack)-1]
				if len(stack) != 0 {
					top = stack[len(stack)-1]
				}
			}
			stack = append(stack, i)
		}
	}
	return res
}
```
#### 总结：
- 单调栈就是栈中数据是单调的
	- 单调增：越深越大，如果当前元素 <  = 栈顶才能压入，用于寻找右边第一个 > 当前元素的元素
- 用一个结果列表表示最终结果
### 题号：496 nextGreaterElement
#### 思路：
- 这个题可以理解为：
	- 从nums2中抽取一部分元素作为nums1，然后找nums1中每个元素在nums2中的下一个更大的值
- 解决方法：
	- 像上一题一样，对nums2中每个元素找右边第一个更大的元素
	- 用一个hash表记录nums1中的元素，每次在nums2中遍历到一个元素就用hash表看看看看它在nums1中，是的话就找到了nums1中元素的下一个更大值
		- 所以nums1的hash可以这么设计：k是元素本身的值，v是元素在nums1中的位置，这样nums2中判断的时候就可以根据位置直接填了
#### c++实现：
```c++

```
#### go实现：
```go
func nextGreaterElement(nums1 []int, nums2 []int) []int {
	res := make([]int, len(nums1))
	for i := 0; i < len(res); i++ {
		res[i] = -1
	}

	hashMap := make(map[int]int, len(nums1))
	for k, v := range nums1 {
		hashMap[v] = k
	}

	stack := []int{0}
	for i := 1; i < len(nums2); i++ {
		top := stack[len(stack)-1]

		if nums2[i] <= nums2[top] {
			stack = append(stack, i)
		} else {
			for len(stack) > 0 && nums2[i] > nums2[top] {
				if val, exists := hashMap[nums2[top]]; exists { // 是否在nums1中有这个值
					res[val] = nums2[i]
				}
				stack = stack[:len(stack)-1]
				if len(stack) > 0 {
					top = stack[len(stack)-1]
				}
			}
			stack = append(stack, i)
		}
	}
	return res
}
```
#### 总结：
- 就是单调栈，小心一点，写的时候对res 的初始化竟然变成修改nums1了，调了半天，离谱
### 题号：503 nextGreaterElements
#### 思路：
- 将两个数组拼接，然后遍历找这个新数组中每一个元素的下一个更大元素，最后裁剪结果
#### c++实现：
```c++

```
#### go实现：
```go
func nextGreaterElements(nums []int) []int {
	dNums := append(nums, nums...)

	res := make([]int, len(dNums))
	for i := 0; i < len(dNums); i++ {
		res[i] = -1
	}

	stack := []int{0}
	for i := 1; i < len(dNums); i++ {
		top := stack[len(stack)-1]

		if dNums[i] <= dNums[top] {
			stack = append(stack, i)
		} else {
			for len(stack) > 0 && dNums[i] > dNums[top] {
				res[top] = dNums[i]
				stack = stack[:len(stack)-1]
				if len(stack) > 0 {
					top = stack[len(stack)-1]
				}
			}
			stack = append(stack, i)
		}
	}
	return res[:len(nums)]
}
```
#### 总结：
- 
### 题号：42 trap
#### 思路：
- 暴力法：
	- 对每列，求出它左边最高的列lh，和右边最高的列rh，取lh和rh的最小值，然后减去当前列的高度ch，就得到当前列能接水的高度
	- 注意第一列和最后一列不接水
- 双指针法优化暴力法：
	- 用一个数组记录每一个位置左边最高高度、一个数组记录每一个位置右边最高高度，那么可以由前一个位置的左右最高高度得出当前位置的左右最高高度，类似于动态规划，关系如下
		- 当前左边最高高度=前一个元素的左边最高高度与当前元素的高度的最大值
		- 右边=后一个元素的右边最高高度与当前元素的高度的最大值
		- 这只需要两个单轮循环就行
- 单调栈：
	- 按行来计算
	- 单调递增栈：越深越大
	- 如果当前元素 > 栈顶元素，就说明出现凹槽了，凹槽的右边是当前元素，左边是栈顶的下一个元素
	- 当前元素 = 栈顶元素，用当前元素下标替换栈顶元素
	- 当前元素  < 栈顶元素，直接压栈
#### c++实现：
```c++

```
#### go实现：
```go
func trap(height []int) int {
	if len(height) <= 2 {
		return 0
	}

	sum := 0
	stack := []int{0}
	for i := 1; i < len(height); i++ {
		top := stack[len(stack)-1] //  栈顶元素
		if height[i] < height[top] {
			stack = append(stack, i)
		} else if height[i] == height[top] {
			stack[len(stack)-1] = i // 相等时栈顶元素替换成当前的下标// 这样可以让栈顶的左边一个是左侧最大值
		} else {
			for len(stack) > 0 && height[i] > height[top] {
				curHeight := height[top]     // 当前元素 > 栈顶元素，说明出现凹槽，凹槽的离地面的高度 = 当前栈顶元素的高度
				stack = stack[:len(stack)-1] // 弹出栈顶元素
				if len(stack) > 0 {
					top = stack[len(stack)-1]                            // 因为弹出了栈顶，现在的栈顶是第二个元素了，也就是左侧高度
					leftHeight := height[top]                            // 左侧高度
					areaHeight := min(leftHeight, height[i]) - curHeight // 凹槽高度 = 左右高度的最小值减去当前离地面高度
					areaWidth := i - top - 1                             // 凹槽宽度 = 当前下标 - 栈顶元素（左侧高度的下标）

					curArea := areaHeight * areaWidth
					sum += curArea
				}
			}
			stack = append(stack, i)
		}
	}
	return sum
}
```
#### 总结：
- 变量的含义要搞清楚，命名要清晰
### 题号：84 largestRectangleArea
#### 思路：
- 找到每个柱子curCo两边第一个比curCo小的柱子smCo，停下来在smCo之前计算
- 用单调递减栈：越深越小，当前元素  <  栈顶元素才计算
	- 需要再柱子首尾加上0
		- 首部0：防止前面两个就是降序的时候，把第一个元素弹出希望取栈的第二个元素导致栈空跳过了计算面积过程
		- 尾部0：防止全升序不会计算面积的情况
	- 
#### c++实现：
```c++

```
#### go实现：
```go
func largestRectangleArea(heights []int) int {
	maxArea := 0
	heights = append(heights, 0)
	heights = append([]int{0}, heights...)

	stack := []int{0}
	for i := 1; i < len(heights); i++ {
		topValue := stack[len(stack)-1]
		if heights[i] > heights[topValue] {
			stack = append(stack, i)
		} else if heights[i] == heights[topValue] {
			stack[len(stack)-1] = i
		} else {
			for heights[stack[len(stack)-1]] > heights[i] {
				mid := stack[len(stack)-1]
				stack = stack[0 : len(stack)-1]
				left := stack[len(stack)-1]
				tmp := heights[mid] * (i - left - 1)
				fmt.Println(tmp, left, i-left-1, mid)
				if tmp > maxArea {
					maxArea = tmp
				}
			}
			stack = append(stack, i)
		}
	}
	return maxArea
}
```
#### 总结：
- 写好一个处理逻辑之后想一想有没有当前处理逻辑处理失效的情况