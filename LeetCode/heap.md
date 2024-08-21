
### 题号：215 最大的第K个数
#### 思路：
- 堆是一个完全二叉树，如果把完全二叉树按层序遍历编号(以下从0开始）：
	- 第i个节点的左子节点是$2i+1$ ，右子节点是$2i+2$ ，父节点$\frac{i-1}{2}$ 
- 大根堆：根节点 > 左右孩子节点，小根堆反之
- 构建大根堆
	- 调整：使得父节点总是 > 孩子节点， 从第$\frac{n}{2}-1$ 个节点开始，到第一个节点结束
		- 如果父节点就是最大，不用调整
		- 如果左子节点最大，交换左子节点和父节点，然后还要把左子节点当做父节点往下处理，因为交换后左子树未必是大根堆
		- 右子节点同样
- 排序
#### c++实现
```c++
#include<vector>
#include<iostream>
using namespace std;

void showVex(vector<int> &nums){
	for (int num : nums){
		cout << num << '\t';
	}
	cout<<endl;
}
/* 建堆 */
void heapify(vector<int> &nums,  int end){
	cout <<"end:"<< end<< endl;
	int beginHeapifyPos = end/2 - 1;
	// showVex(nums);
	int i;
	for (i = beginHeapifyPos; i >= 0; i--){
		int maxChild = i*2 + 1;
		while((i*2+1) <= end){
			// showVex(nums);
			if (maxChild + 1 <= end && nums[maxChild+1]>nums[maxChild]){
				maxChild+=1;
			}
			if (nums[maxChild]>nums[i]){
				swap(nums[maxChild], nums[i]);
				showVex(nums);
				i = maxChild;
				cout << "i:" <<i<< endl;
				maxChild = 2*maxChild+1;
			}else{
				break;
			}
		}
	}
}

int findKthLargest(vector<int>& nums, int k) {
	int i;
	int n = nums.size();
	for (i=0; i<k;i++){
		heapify(nums, n-1-i);
		showVex(nums);
		cout << endl;
		swap(nums[0], nums[n-1-i]);
		showVex(nums);
	}
	return nums[n-k];
}

int main(){
	vector<int> nums = {3, 2, 1, 5, 6, 4};
	cout << findKthLargest(nums, 2);
}
```