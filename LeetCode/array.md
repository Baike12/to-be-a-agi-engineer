### 题号：704 search
#### 思路：
- 都在闭区间里面，思路清晰
#### c++实现：
```c++
int search(vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;
    int mid;
    while (left <= right)
    {
        mid = left + ((right - left)>>1);   
        if(nums[mid] > target)
        {
            right = mid - 1;
        }else if (nums[mid] < target)
        {
            left = left + 1;    
        }else 
        {
            return mid;
        }
    }
    return -1;
}
```
#### go实现：
```go

```
### 题号：27 removeElement
#### 思路：
- 快慢指针
	- 快指针扫描整个数组，遇到不等于val的就覆盖慢指针的指向的元素，然后慢指针后移一位
	- 遇到等于val的不覆盖，慢指针也不移动
	- 这样就相当于慢指针保留了不等于val的元素
#### c++实现：
```c++
int removeElement(vector<int>& nums, int val) {
    int slow = 0;
    int fast = 0;
    for (fast;fast < nums.size();fast++){
        if (nums[fast] != val){
            nums[slow] = nums[fast];
            slow++;
        }
    }
    return slow;
}
```
#### go实现：
```go

```
### 题号：977 sortedSquares
#### 思路：
- 双指针
	- 两个指针分别指向首尾，取两个值的平方中大的一个放到新数组中
#### c++实现：
```c++
vector<int> sortedSquares(vector<int>& nums) {
    vector<int> res(nums.size(), 0);
    int resPos = nums.size()-1;
    for (int i=0, j=nums.size()-1 ; i <= j;){
        if (nums[i] * nums[i] > nums[j]*nums[j]){
            res[resPos] = nums[i] * nums[i];
            resPos--;
            i++;
        }else{
            res[resPos] = nums[j] * nums[j];
            resPos--;
            j--;
        }
    }
    return res;
}
```
#### go实现：
```go

```
### 题号：209 minSubArrayLen
#### 思路：
- 滑动窗口双指针
	- 右指针右移直到窗口内的值 > = 目标值
	- 然后右移左指针缩小窗口，记录当前长度curlen
	- 重复以上过程当curlen < 上一轮的curlen时更新curlen
#### c++实现：
```c++
int minSubArrayLen(int target, vector<int>& nums) {
    int res = INT32_MAX;// 后面curlen比res小就用curlen代替它，所以res初始化为最大值
    int curLen;
    int sum = 0;
    int l = 0;
    for (int r = 0; r < nums.size(); r++){
        sum += nums[r];
        while (sum >= target)
        {
            curLen = r - l +  1;
            res = curLen < res? curLen:res;
            sum -= nums[l];
            l++;
        }
    }
    return res == INT32_MAX?0:res;
}
```
#### go实现：
```go

```
### 题号：59 generateMatrix
#### 思路：
- 画圈
	- 由外到内，一圈一圈画，关键是**左闭右开**（也可以是其他的，但是最好一致，这样可以同一的逻辑处理）
- 注意：每循环一周更新行和列
#### c++实现：
```c++
void show(vector<vector<int> > res){
    for (int i = 0; i < res.size(); i++){
        for (int j = 0; j < res[0].size(); j++){
            cout << res[i][j] << "\t" ;
        }
        cout << endl;
    }
    cout << endl;
}

vector<vector<int> > generateMatrix(int n) {
    vector<vector<int> > res(n, vector<int>(n, 0));
    int curCircle = 0;
    int row = 0;
    int col = 0;
    int count = 1;

    while (curCircle < n / 2)
    {   
        row = curCircle;
        col = curCircle;
        cout << "curCircle:" << curCircle << endl;
        for (col + curCircle; col < n - 1 - curCircle; col++){
            res[row][col] = count++;
        }
        show(res);
        for (row + curCircle; row < n - 1 - curCircle; row++){
            res[row][col] = count++;
        }
        show(res);
        for (col - curCircle; col > curCircle; col--){
            res[row][col] = count++;
        }
        show(res);
        for (row - curCircle; row > curCircle; row--){
            res[row][col] = count++;
        }
        show(res);
        curCircle++;
    }
    if (n % 2 == 1){
        res[curCircle][curCircle] = count;
    }
    return res;
}
```
#### go实现：
```go

```