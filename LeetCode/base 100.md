### 题号：字母异位词
#### 思路：
- 将每一个字符串内部排序，异位词就会排到一起
- 然后用一个键string值string切片的hash保存一组异位词
#### c++实现：
```c++

```
#### go实现：
```go
func groupAnagrams(strs []string) [][]string {
	hash := make(map[string][]string)

	for _, str := range strs {
		bstr := []byte(str)
		sort.Slice(bstr, func(i, j int) bool { return bstr[i] < bstr[j] })
		sortedStr := string(bstr)
		hash[sortedStr] = append(hash[sortedStr], str)
	}

	res := make([][]string, 0, len(hash))
	for _, resSlice := range hash {
		res = append(res, resSlice)
	}
	return res
}
```
#### 总结：
- 使用make的时候如果知道容量尽量把容量写上去
### 题号：最长连续序列
#### 思路：
- 用一个hash表保存所有数
- 遍历所有数，当一个数的值 - 1不在hash表，就从这个数开始，当然，只有一个数的值-1不在hash表，才从它开始，因为如果它的值 - 1在hash表，从它的值 - 1开始能得到更长的序列
#### c++实现：
```c++

```
#### go实现：
```go
func longestConsecutive(nums []int) int {
	set := make(map[int]bool, len(nums))
	for _, num := range nums {
		set[num] = true
	}

	maxLen := 0
	for num := range set {
		if !set[num-1] {
			curLen := 1
			curNum := num
			for set[curNum+1] {
				curLen++
				curNum++
			}
			maxLen = max(curLen, maxLen)
		}
	}
	return maxLen
}
```
#### 总结：
- 其实也不是严格O(n)但是能过
### 题号：移动0
#### 思路：
- 快慢指针，快指针遇到非0就移动到慢指针上
#### c++实现：
```c++

```
#### go实现：
```go
func moveZeroes(nums []int) {
	slow, fast := 0, 0
	zeroNum := 0

	for ; fast<len(nums); fast++{
		if nums[fast] != 0{
			nums[slow] = nums[fast]
			slow++
		}else{
            zeroNum++
        }
	}

	for i:=1;i<=zeroNum;i++{
		nums[len(nums)-i] = 0
	}
}

```
#### 总结：
- 
### 题号：容下最多的水
#### 思路：
- ~~单调栈，用一个变量记录最多的水~~
- 双指针更简单
#### c++实现：
```c++

```
#### go实现：
```go
func maxArea(height []int) int {
    mA := 0
    left, right := 0, len(height)-1

    for left < right{
        mA = max(min(height[left] , height[right]) * (right-left), mA)
        if height[left] <= height[right]{
            left++
        }else{
            right--
        }
    }
    return mA
}
```
#### 总结：
- 
### 题号：三数之和
#### 思路：
- 关键点在于避免重复的结果和降低复杂度，这里最多也只能降到$O(n^{2})$ 
- 先排序，排序能让一样大的数聚集在一起，然后遍历的时候只要考虑当前值是否和前一个值相同就行了
- 选第二第三个数的时候使用双指针，缩成一轮循环
#### c++实现：
```c++

```
#### go实现：
```go
func threeSum(nums []int) [][]int {
	sort.Ints(nums)
	res := make([][]int, 0)

	for i := 0; i < len(nums); i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}
		target := -nums[i]
		secondPos := i + 1
		thirfPos := len(nums) - 1
		for ; secondPos < len(nums); secondPos++ {
			if secondPos > i+1 && nums[secondPos] == nums[secondPos-1] {
				continue
			}

			for thirfPos > secondPos && nums[thirfPos]+nums[secondPos] > target {
				thirfPos--
			}

			if secondPos == thirfPos { // cur nums does not fit
				break
			}
			if nums[secondPos]+nums[thirfPos]==target{
				res = append(res, []int{nums[i], nums[secondPos], nums[thirfPos]})
			}
		}
	}
	return res
}
```
#### 总结：
- 在第二层循环中，当第二个数加上第三个数 > 目标值时，说明第三个数太大了，直到不大于，此时可能 < 或 = ，如果第二个数下标和第三个数下标都一样大了和还是偏大，说明当前第二个数作为第二个数不可能满足条件了，直接退出第二个数的循环
### 题号：无重复字符的最大子串
#### 思路：
- 滑动窗口 + hash
- 遍历字符串，将字符加入到窗口中，如果新加入的字符在窗口中已经有了，就收窄窗口左侧
#### c++实现：
```c++

```
#### go实现：
```go
func lengthOfLongestSubstring(s string) int {
	res := 0
	left, right := 0, 0
	hash := make(map[byte]bool)

	for right < len(s) {
		for len(hash) > 0 && hash[s[right]] {		
			delete(hash, s[left])
            left++
		}
		hash[s[right]] = true       
		res = max(res, right - left +1)
        right++
        fmt.Println(right, left)
	}
	return res
}
```
#### 总结：
- 一定要注意下标什么时候 +  + 
- 左侧下标应该是移除之后 + 
- 右侧下标在计算结果之后 + ，不然计算的结果会多1
### 题号：和为k的子数组
#### 思路：
- 用一个pre记录第一个到第i的所有数的和
- 用一个hash表记录一路上出现的前缀和出现的次数，因为有正有负，一个前缀和可能出现不止一次
#### c++实现：
```c++

```
#### go实现：
```go
func subarraySum(nums []int, k int) int {
	prefixSum := 0
	numOfSumK := 0
	hash := make(map[int]int)
    hash[0] = 1

	for i := 0; i < len(nums); i++ {
		prefixSum += nums[i]
		if count, exists := hash[prefixSum-k]; exists {
			numOfSumK += count
		}
		hash[prefixSum]++
	}
	return numOfSumK
}
```
#### 总结：
- 记得把0出现了一次加上
### 题号：除自身以外数组的乘积
#### 思路：
- 一个表示从左到右的乘积的，一个从右到左的乘积
- 当前的值 = 从左到右的前一个下标值 × 从右到左的后一个下标的值
- 然后首尾元素单独设置
#### c++实现：
```c++

```
#### go实现：
```go
func productExceptSelf(nums []int) []int {
	fromHeadToTail := []int{}
	htt := 1
	for i := 0; i < len(nums); i++ {
		htt = htt * nums[i]
        fmt.Println(htt)
		fromHeadToTail = append(fromHeadToTail, htt)
	}

	fromTailToHead := make([]int, len(nums))
	tth := 1
	for j := len(nums)-1; j >=0; j--{
        fmt.Println(tth)
		tth = tth*nums[j]
		fromTailToHead[j] = tth
	}

	res := make([]int,len(nums))
	res[0] = fromTailToHead[1]
	res[len(nums)-1] = fromHeadToTail[len(nums)-2]
	for i :=1;i<len(nums)-1;i++{
		res[i] = fromHeadToTail[i-1] * fromTailToHead[i+1]
	}
	return res
}
```
#### 总结：
- 
### 题号：滑动窗口最大值
#### 思路：
- 用一个大根堆保存窗口内的值
- 插入之后，如果堆顶元素也就是最大值不在窗口内，要弹出，因为是先压入再处理弹出，所以要循环的弹出而不是只弹一次
#### c++实现：
```c++

```
#### go实现：
```go
var b []int

type hp1 struct {
	sort.IntSlice
}

func (h hp1) Less(i, j int) bool {
	return b[h.IntSlice[i]] > b[h.IntSlice[j]]
}

func (h *hp1) Push(v interface{}) {
	h.IntSlice = append(h.IntSlice, v.(int))
}

func (h *hp1) Pop() interface{} {
	bb := h.IntSlice
	v := h.IntSlice[len(h.IntSlice)-1]
	h.IntSlice = bb[:len(bb)-1]
	return v
}

func maxSlidingWindow1(nums []int, k int) []int {
	b = nums
	q := &hp1{make([]int, k)}
	for i := 0; i < k; i++ {
		q.IntSlice = append(q.IntSlice, i)
	}
	heap.Init(q)

	res := make([]int, len(nums)-k+1)
	res[0] = b[q.IntSlice[0]]

	for i := k; i < len(nums); i++ {
		q.IntSlice = append(q.IntSlice, i)
		for q.IntSlice[0] <= i-k {
			q.Pop()
		}
		res = append(res, b[q.IntSlice[0]])
	}
	return res
}

```
#### 总结：
- go实现堆
	- 定义一个结构体t包含一个排序接口is
	- t实现is的Less功能
	- 然后t本身实现heap的接口Push和Pop
	- 最后实例化一个t后用heap.init初始化堆