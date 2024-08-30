### 题号：215 kth value:fast sort
#### 思路：
- 找第k大的元素，快排一般按从小到大的顺序，所以是n - k + 1小的元素
- 快速排序
	- 使用最左边的的元素作为pivot
	- 移动比pivot大的元素到右边，比pivot小的到左边
	- 移动pivot元素到分界为止
- 这里实际上使用快速选择，减少递归次数
	- 一轮排序之后，此轮排序中选中的pivot位置也就是最终位置，只要这轮排序中的pivot的位置pp是第k大的元素所在位置kp，直接返回这个pivot的位置就行了
	- 如果kp  < pp，只要在pp的左侧排序就行，同理kp > pp在右侧排序
- 具体实现：
	- 
#### c++实现：
```c++
class Solution{
    public:
    int res;
    int partSort(vector<int> &nums, int left, int right){
        int initPivotPos = left;
        int pivot = nums[left];
        while(left < right){
            while(left < right && nums[right] >= pivot){
                right--;
            }
            while(left < right && nums[left] <= pivot){
                left++;
            }
            swap(nums[left], nums[right]);
        }
        swap(nums[initPivotPos], nums[right]);
        return right;
    }

    void quickselect(vector<int>&nums, int left, int right, int k){
        if (left == right){
            if (k == left){ // 
                res = nums[k];
            }
            return;
        }
        int pivotPos = partSort(nums, left, right);
        if (pivotPos == k){
            res = nums[pivotPos];
        }else if(pivotPos > k){
            quickselect(nums, left, pivotPos - 1, k);
        }else if(pivotPos < k){
            quickselect(nums, pivotPos + 1, right, k);
        }
        return;
    }
    int findKthLargest(vector<int> &nums, int k) {
        int n = nums.size();
        quickselect(nums, 0, n - 1, n - k);
        return res;
    }
};
```
#### go实现：
```go

```

