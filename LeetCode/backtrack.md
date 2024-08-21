### 题号：77 组合
#### 思路：
- 回溯用来解决嵌套层数问题
- 用一个start记录当前层的其实位置
#### c++实现：
```c++
class Solution {
public:
	vector<vector<int>> res;
	vector<int> path;
	void backtrack(int n, int k, int start){
		if(path.size() == k){
			res.push_back(path);
			return;
		}
		for (int i=start; i<=n;i++){
			path.push_back(i);
			backtrack(n, k, i+1);
			path.pop_back();
		}
	}
    vector<vector<int>> combine(int n, int k) {
		backtrack(n,k,1);
		return res;
    }
};
```
#### go实现：
```go

```
#### 优化：剪枝
- 在每一层如果起始位置之后的元素个数已经不够组合的个数，就不用遍历了
```c++
class Solution {
public:
	vector<vector<int>> res;
	vector<int> path;
	void backtrack(int n, int k, int start){
		if(path.size() == k){
			res.push_back(path);
			return;
		}
		for (int i=start; i<=n-(k-path.size())+1;i++){
			path.push_back(i);
			backtrack(n, k, i+1);
			path.pop_back();
		}
	}
    vector<vector<int>> combine(int n, int k) {
		backtrack(n,k,1);
		return res;
    }
};
```
### 题号：216 组合总和III
#### 思路：
- 要找k个，当path中有k个返回，如果时path中的值的和等于n，将path加入到结果中
- 剪枝：如果剩下要遍历的个数都不够需要的个数，就不用遍历的
#### c++实现：
```c++
class Solution {
public:
	vector<vector<int>> res;
	vector<int> path;
	void backtrack(int curTarget, int start, int k){
		if (path.size() == k){
			if (curTarget == 0){
				res.push_back(path);
			}
			return;
		}
		for (int i=start; i<=9-(k-path.size())+1; i++){
			path.push_back(i);
			backtrack(curTarget-i, i+1,k);
			path.pop_back();
		}
	}
    vector<vector<int>> combinationSum3(int k, int n) {
		backtrack(n, 1,k);
		return res;
    }
};
```
#### go实现：
```go

```
### 题号：17 letterCombinations
#### 思路：
- 终止条件：输入的数字的个数
- 用一个index来记录当前遍历到输入数字中的第几个了
- 每一个数字对应一个字符串，递归到这一层的时候遍历这个字符串取出字符
#### c++实现：
```c++
class Solution {
public:
	vector<string> res;
	string s;
	const string letter[10] = {
		"", // 0
        "", // 1
        "abc", // 2
        "def", // 3
        "ghi", // 4
        "jkl", // 5
        "mno", // 6
        "pqrs", // 7
        "tuv", // 8
        "wxyz", // 9
	};
	void backtrack(const string &digits, int start){
		if (start == digits.size()){
			res.push_back(s);
			return;
		}
		string curLetter = letter[digits[start]-'0'];
		for (int i = 0; i<curLetter.size();i++){
			s.push_back(curLetter[i]);
			backtrack(digits, start+1);
			s.pop_back();
		}
	}
    vector<string> letterCombinations(string digits) {
        if (digits == "")return vector<string>();
		backtrack(digits, 0);
		return res;
    }
};
```
#### go实现：
```go

```