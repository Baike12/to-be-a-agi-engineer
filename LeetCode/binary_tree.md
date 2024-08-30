### 题号：144二叉树前序遍历
#### 思路：
- 递归前序遍历
- 对于递归，前中后序实现上区别不大
#### c++实现：
``` C++
struct TreeNode
{
	int val;
	TreeNode *left;
	TreeNode *right;
}

class Solution
{
public:
	/* 使用递归遍历，要在函数中调用函数，所有要重新写一个递归函数 */
	void traversal(TreeNode *cur, vector<int> &vec)
	{
		if (cur == nullptr)
		{
			return;
		}
		vec.push_back(cur->val);
		traversal(cur->left, vec);
		traversal(cur->right, vec);
	}
	vector<int> preorderTraversal(TreeNode *root)
	{
		vector<int> vec;
		traversal(root, vec);
		return vec;
	}
}
```
#### go实现：
```go

```

### 题号：144 前序遍历（迭代法）
#### 思路：
- 访问顺序为中左右，入栈顺序为中右左
- 判断根节点是否为空，为空直接返回
- 根节点入栈
- 处理栈顶
- 右孩子入栈，左孩子入栈
- 下一轮循环，这时左孩子在栈顶，一直循环直到栈空
	- 每次循环能处理一个节点，从根节点开始
#### c++实现：
```c++
class Solution
{
public:
	vector<int> preorderTraversal(TreeNode *root)
	{
		stack<TreeNode *> st;
		vector<int> vt;
		if (root == nullptr)
		{
			return vt;
		}
		st.push(root);
		while(!st.empty()){
			TreeNode* node = st.top();
			vt.push_back(node->val);
			st.pop();
			if(node->right != nullptr) st.push(node->right);
			if(node->left != nullptr) st.push(node->left);
		}
		return vt;
	}
};
```
#### go实现：
```go

```
### 题号：94 中序遍历
#### 思路：
- 访问顺序是左中右，需要往左子节点一直访问直到为空
- 访问的节点和处理的节点不是同一个
- 用一个指针cur指向当前访问的节点，用一个栈记录访问过的节点
- 一开始cur指向根节点
- 当cur非空时，将cur压入栈，然后将cur指向cur的的左子节点
- cur空时，弹出栈顶元素，并将栈顶元素的值放到返回序列中，然后cur指向栈顶元素的右子节点
	- 说明上一个cur没有左子节点，处理上一个访问的节点，栈顶的元素，处理完后cur指向cur的右子节点
	- 说明上一个cur左右子节点都没有，处理上上一个访问的节点，也就是栈顶元素
#### c++实现：
```c++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
		stack<TreeNode*> st;
		vector<int> vt;
		TreeNode *cur = root;
		while(cur != nullptr || !st.empty()){
			if (cur != nullptr){
				st.push(cur);
				cur = cur->left;
			}else{
				cur = st.top();
				vt.push_back(cur->val);
				st.pop();
				cur = cur->right;
			}
		}
		return vt;
    }
};
```
#### go实现：
```go

```


### 题号：145 后序遍历
#### 思路：
- 遍历顺序为左右中
- 前序遍历为中左右，调整为中右左，然后将输出逆序
#### c++实现：
```c++
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
		stack<TreeNode *> st;
		vector<int> vt;
		if (root == nullptr){
			return vt;
		}
		TreeNode * cur;
		st.push(root);
		while(!st.empty()){
			cur = st.top();
			vt.push_back(cur->val);
			st.pop();
			if (cur->left != nullptr) st.push(cur->left);
			if (cur->right != nullptr) st.push(cur->right);
		}
		reverse(vt.begin(), vt.end());
		return vt;
    }
};
```
#### go实现：
```go

```

### 题号：102 层序遍历
#### 思路：
- 使用队列
	- 在每一层将下一层的所有节点入队，当前层已经全部入队了，只需要遍历队列大小
- 递归
	- 参数：当前节点、结果二维向量，当前深度
	- 返回条件：当前节点为空
	- 如果当前结果的大小等于，注意这里的大小是第0维的大小，用第0维的索引来表示深度
#### c++实现：
```c++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
		vector<vector<int>> vec;
		if (root == nullptr)return vec;
		queue<TreeNode*> que;
		que.push(root);
		while (!que.empty()){
			vector<int> subVec;
			int size = que.size();
			int i = 0;
			for  (i=0; i<size; i++){
				TreeNode *cur = que.front();
				subVec.push_back(cur->val);
				que.pop();
				if (cur->left != nullptr) que.push(cur->left);
				if (cur->right != nullptr) que.push(cur->right);
			}
			vec.push_back(subVec);
		}
		return vec;
    }
};

// 递归实现
class Solution {
public:
	void traversal(vector<vector<int>> &result, TreeNode *cur, int depth){
		if (cur == nullptr)return;
		if (result.size() == depth) result.push_back(vector<int>());
		result[depth].push_back(cur->val);
		traversal(result, cur->left, depth+1);
		traversal(result, cur->right, depth+1);
	}
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
		int depth = 0;
		vector<vector<int>> result;
		traversal(result, root, depth);
		return result;
    }
};
```
#### go实现：
```go

```


### 题号：637 层平均值
#### 思路：
- 
#### c++实现：
```c++
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        queue<TreeNode *> que;
        vector<double> res;
        if (root == nullptr)return res;
        que.push(root);
        while(!que.empty()){
            TreeNode *cur;
            int size = que.size();
            int i;
            double sum=0;
            for (i=0;i<size;i++){
                cur = que.front();
                sum += cur->val;
                que.pop();
                if (cur->left != nullptr) que.push(cur->left);
                if (cur->right != nullptr) que.push(cur->right);
            }
            res.push_back(sum/size);
        }
        return res;
    }
};
```
#### go实现：
```go

```
### 题号：116 填充next指针
#### 思路：
- 
#### c++实现：
```c++

```
#### go实现：
```go
class Solution {
public:
    Node* connect(Node* root) {
		queue<Node *> que;
		if (root == nullptr)return root;
		que.push(root);
		while (!que.empty()){
			int size = que.size();
			int i;
			Node *cur;
			Node *ne;
			for (i = 0; i< size; i++){
				cur = que.front();
				que.pop();
				if (!que.empty()){ 
                    ne = que.front();
                }else{
                    ne = nullptr;
                }
                cout << "cur val: " << cur->val << endl;
                if (ne != nullptr) cout << "ne val: " << ne->val << endl;
                cout << "----------" << endl;
				if (i == size - 1){
					cur->next == nullptr;
                    cout << "cur val tail: " << cur->val << endl;
				}else{
					cur->next = ne;
                    cout << "cur val: " << cur->val << endl;
                    cout << "cur next val: " << cur->next->val << endl;
				}
                cout << "+++++++++++" << endl;
				if (cur->left != nullptr) que.push(cur->left);
				if (cur->right != nullptr) que.push(cur->right);
			}
		}
		return root;
    }
};
```


### 题号：111 最小深度
#### 思路：
- 需要注意的点是只有当左右子节点都为空才是叶子节点 
#### c++实现：
```c++
class Solution {
public:
    int minDepth(TreeNode* root) {
		queue<TreeNode*> q;
		int depth = 0;
		if (root == nullptr)return depth;
		q.push(root);
		while (!q.empty()){
			depth++;
			int size = q.size();
			int i;
			TreeNode * cur;
			for (i = 0; i< size;i++){
				cur = q.front();
				q.pop();
				if (cur->left != nullptr) q.push(cur->left);
				if (cur->right != nullptr) q.push(cur->right);
				if (cur->left == nullptr && cur->right == nullptr)return depth;
			}
		}
		return depth;
    }
};
```
#### go实现：
```go

```


### 题号：226 翻转二叉树
#### 思路：
- 每个节点都翻转，整体就翻转了
- 注意中序遍历不行 
#### c++实现：
```c++
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
		if (root == nullptr)return root;
		swap(root->left, root->right);
		invertTree(root->left);
		invertTree(root->right);
		return root;
    }
};
```
#### go实现：
```go

```

### 题号：101 对称二叉树
#### 思路：
- 参数：要比较的两个节点，左节点和右节点
- 递归终止条件：两个节点是否相等，返回终止时的结果
	- 分为空和非空
- 递归：比较左节点的左子节点，右节点的右子节点
#### c++实现：
```c++
class Solution {
public:
	bool traversal(TreeNode *l, TreeNode *r){
		/* 只有对称才能继续递归：也就是左右节点都存在并且值相等 */
		if (l == nullptr && r != nullptr){
			return false;
		}else if (l != nullptr && r == nullptr){
			return false;
		}else if (l == nullptr && r == nullptr){
			return true;
		}else if (l->val != r->val){
			return false;
		}

		bool outside = traversal(l->left, r->right);/* 需要从下一层返回消息，所以需要返回值 */
		bool inside = traversal(l->right, r->left);
		return outside && inside;
	}
    bool isSymmetric(TreeNode* root) {
		bool isSym = traversal(root->left, root->right);
		return isSym;
    }
};
```
#### go实现：
```go

```


### 题号：104 最大深度
#### 思路：
- 后序遍历：先从左右子节点获取深度，取他们中最大的 + 1返回给上一层
- 参数：传入一个节点；返回当前节点的高度
#### c++实现：
```c++
class Solution {
public:
	int traversal(TreeNode *node){
		if (node==nullptr){
			return 0;
		}
		int leftHight = traversal(node->left);
		int rightHight = traversal(node->right);
		return 1 + max(leftHight, rightHight);
	}
    int maxDepth(TreeNode* root) {
		if (root == nullptr)return 0;
		return traversal(root);
    }
};
```
#### go实现：
```go

```
### 题号：111 最小深度
#### 思路：
- 后序遍历
	- 当前节点为空返回
	- 递归左右子树获取深度
	- 处理当前节点
		- 左空右不空：返回1 + 右子树深度
		- 右空左不空：返回1 + 左子树深度
		- 左右都空：返回1
		- 左右都不空：返回1 + 左子树深度和右子树深度的最小值
- 前序遍历
	- 用一个初始化为最大值的全局变量res保存当前深度，递归函数就不返回值了
	- 传入深度de记录当前到第几层了
	- 处理当前节点：
		- 如果当前节点是叶子节点，用de和res的最小值更新res
#### c++实现：
```c++
// 后序遍历
class Solution {
public:
	int traversal(TreeNode *node){
		if (node == nullptr){
			return 0;
		}
		int leftHight = traversal(node->left);
		int rightHight = traversal(node->right);

		if (node->left != nullptr && node->right == nullptr){
			return 1+leftHight;
		}
		if (node->left == nullptr && node->right != nullptr){
			return 1+rightHight;
		}
		return 1 + min(leftHight, rightHight);
	}
    int minDepth(TreeNode* root) {
		if (root == nullptr)return 0;
		return traversal(root);
    }
};


// 前序遍历
class Solution {
private:
    int result;
    void getdepth(TreeNode* node, int depth) {
        // 函数递归终止条件
        if (node == nullptr) {
            return;
        }
        // 中，处理逻辑：判断是不是叶子结点
        if (node -> left == nullptr && node->right == nullptr) {
            result = min(result, depth);
        }
        if (node->left) { // 左
            getdepth(node->left, depth + 1);
        }
        if (node->right) { // 右
            getdepth(node->right, depth + 1);
        }
        return ;
    }

public:
    int minDepth(TreeNode* root) {
        if (root == nullptr) {
            return 0;
        }
        result = INT_MAX;
        getdepth(root, 1);
        return result;
    }
};
```
#### go实现：
```go

```
### 题号：222 完全二叉树节点个数
#### 思路：
- 是否满二叉树：一直往左的深度等于已知往右的深度
	- 是：节点个数=$2^{h}-1$ 
	- 不是：递归左右子树，返回左右子树的节点个数 + 1
#### c++实现：
```c++
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (root == nullptr)return 0;
		int lD=0;
        int rD = 0;
		TreeNode *left = root->left;
		TreeNode *right = root->right;
		while(left){
			lD++;
			left = left->left;
		}
		while(right){
			rD++;
			right = right->right;
		}
		if (lD == rD){
			return (2 << lD)  - 1;
		}
		int leftNum = countNodes(root->left);
		int rightNum = countNodes(root->right);
		return leftNum + rightNum +1;
    }
};
```
#### go实现：
```go

```
### 题号：110 平衡二叉树判断
#### 思路：
- 平衡二叉树的任意两个子树的差值最多为1
- 后序遍历：因为要先获取左右子树的高度
	- 如果左右子树的高度差已经 > 1，返回一个特殊标识符-1
	- 否则返回左右子树的高度
#### c++实现：
```c++
class Solution {
public:
	int traversal(TreeNode *node){
		if (node ==  nullptr){
			return 0;
		}
		int lh = traversal(node->left);
		int rh = traversal(node->right);
		if (lh == -1 || rh == -1){
			return -1;
		}

		if (abs(lh - rh) > 1){
			return -1;
		}else{
			return 1+max(lh, rh);
		}
	}
    bool isBalanced(TreeNode* root) {
		return traversal(root)==-1?false:true;
    }
};
```
#### go实现：
```go

```
### 题号：257 二叉树所有路径
#### 思路：
- 前序遍历
	- 当前节点没有子节点时将字符串添加进结果集
	- 递归处理完一个节点的子节点，返回时要回溯
#### c++实现：
```c++
class Solution {
public:
	vector<string> res;
	vector<int> path;
	void traversal(TreeNode *cur, vector<int> &p, vector<string> &r){
		p.push_back(cur->val);

		if (cur->left == nullptr && cur->right == nullptr){
			string s;
			int size = p.size();
			int i;
			for (i = 0; i<size-1;i++){
				s += to_string(p[i]);
				s += "->";
			}
			s += to_string(p[size-1]);
			res.push_back(s);
		}

		if (cur->left){
			traversal(cur->left, path, res);
			path.pop_back();
		}
		if (cur->right){
			traversal(cur->right, path, res);
			path.pop_back();
		}
	}
    vector<string> binaryTreePaths(TreeNode* root) {
		traversal(root, path, res);
		return res;
    }
};
```
#### go实现：
```go

```
### 题号：404 左叶子之和
#### 思路：
- 当前节点节点无法判断自己是不是左叶子，要用父节点判断
- 后序遍历：要从左右子节点返回和
	- 终止条件：当前节点为空或者为叶子节点（叶子节点没有父节点）
	- 单层：
#### c++实现：
```c++
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
		if (root == nullptr){
			return 0;
		}
		int leftLeafVal = 0;
		if (root->left && !root->left->left && !root->left->right){
			leftLeafVal = root->left->val;
		}
		int leftVal = sumOfLeftLeaves(root->left);
		int rightVal = sumOfLeftLeaves(root->right);
		return leftVal + rightVal + leftLeafVal;// 当前层处理
    }
};
```
#### go实现：
```go

```
### 题号：513 树最左下角的值
#### 思路：
- 层序遍历
#### c++实现：
```c++
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
		int res;
		queue<TreeNode *> q;
		q.push(root);
		while (!q.empty()){
			int size = q.size();
			int i;
			TreeNode *cur;
			for (i = 0; i < size; i++){
				cur = q.front();
				if (i == 0) res = cur->val;
				q.pop();
				if (cur->left)q.push(cur->left);
				if (cur->right)q.push(cur->right);
			}
		}
		return res;
    }
};
```
#### go实现：
```go

```
### 题号：112 路径总和
#### 思路：
- 回溯
	- 用targetSum一路减，如果有减到0就返回真
- 注意：好好读题，一开始没注意是到叶子结点浪费了十分钟~~
#### c++实现：
```c++
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;
		bool lhas = hasPathSum(root->left, targetSum - root->val);
		bool rhas = hasPathSum(root->right, targetSum - root->val);
        if (!root->left && !root->right && root->val == targetSum)return true;
		if (lhas || rhas){
			return true;
		}else{
			return false;
		}
    }
};
```
#### go实现：
```go

```
### 题号：106 从后序遍历和中序遍历构造二叉树
#### 思路：
- 取后序遍历的最后一个节点poev
	- 如果序列长为0直接返回null
	- 如果序列长为1，返回这个值构建的节点
- 切中序遍历序列：在中序遍历中找到poev的位置，从这个位置把中序遍历的序列切成两段，切的过程左闭右开（也可以是其他的，但是要一致）
- 切后序遍历序列：
	- 关键在于后序遍历的长度和中序遍历的一样
	- 当然后序遍历的最后一个节点要删除掉
- 递归的处理中序左子序列和后序左子序列，以及中序右子序列和后序右子序列
#### c++实现：
```c++
class Solution {
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
		if (postorder.size() == 0)return nullptr;
		int postEndVal = postorder[postorder.size()-1];
		int i;
		int postEndValPosInIn;
		TreeNode *root = new TreeNode(postEndVal);
		if (postorder.size() == 1)return root;
		for (i = 0; i < inorder.size(); i++){
			if (inorder[i] == postEndVal){
				postEndValPosInIn = i;
				break;
			}
		}
		/* 拆分中序序列 */
		vector<int>leftInorder(inorder.begin(), inorder.begin()+postEndValPosInIn);
		vector<int>rightInorder(inorder.begin()+postEndValPosInIn+1, inorder.end());

		/* 拆分后序序列 */
		postorder.resize(postorder.size()-1);
		vector<int>leftPostorder(postorder.begin(), postorder.begin()+leftInorder.size());
		vector<int>rightPostorder(postorder.begin()+leftInorder.size(), postorder.end());

		/* 递归 */
		root->left = buildTree(leftInorder, leftPostorder);
		root->right = buildTree(rightInorder, rightPostorder);
		return root;
    }
};
```
#### go实现：
```go

```
### 题号：654 最大二叉树
#### 思路：
- 注意nums只有一节点时直接返回
#### c++实现：
```c++
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
		TreeNode *node = new(TreeNode);
		if (nums.size() == 1){
			node->val = nums[0];
			return node;
		}

		int maxVal = 0;
		int maxValPos = 0;
		int i=0;
		for (i = 0; i<nums.size(); i++){
			if(nums[i] > maxVal){
				maxVal = nums[i];
				maxValPos = i;
			}
		}
		node->val = maxVal;
		/* 当最大值左边或者右边有元素时创建左节点或者右节点 */
		if (maxValPos > 0){
			vector<int> leftVec(nums.begin(), nums.begin()+maxValPos);
			node->left = constructMaximumBinaryTree(leftVec);
		}
		if (maxValPos < nums.size()-1){
			vector<int> rightVex(nums.begin()+maxValPos+1, nums.end());
			node->right = constructMaximumBinaryTree(rightVex);
		}
		return node;
    }
};
```
#### go实现：
```go

```
### 题号：617 合并二叉树
#### 思路：
- 同时遍历两个树
- 复用其中一个树作为基础，下面以复用t1为例
- 递归终止条件：如果一个节点为空，返回另一个节点，都空返回null
- 复用t1修改t1节点的值
- 构建左子节点和右子节点
#### c++实现：
```c++
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
		if (!root1 && root2){
			return root2;
		}
		if (root1 && !root2){
			return root1;
		}
		if(!root1 && !root2){
			return nullptr;
		}
		root1->val += root2->val;
		root1->left = mergeTrees(root1->left, root2->left);
		root1->right = mergeTrees(root1->right, root2->right);
		return root1;
    }
};

class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
		while(root != nullptr){
			if (root->val > val)root = root->left;
			else if(root->val < val) root = root->right;
			else return root;
		}
		return nullptr;
    }
};
```
#### go实现：
```go

```
### 题号：700 二叉搜索树中的搜索
#### 思路：
- 返回子树其实是返回节点
- 搜索树的有序性指引递归方向
	- 在使用迭代法时使得不再需要栈或者队列，因为遍历的方向是确定的
#### c++实现：
```c++
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
		if (root == nullptr){
			return nullptr;
		}
		if (root->val ==  val){
			return root;
		}
		TreeNode* node = nullptr;
		if (root->val > val){
			node = searchBST(root->left, val);
		}
		if (root->val < val){
			node = searchBST(root->right, val);
		}
		return node;
    }
};
```
#### go实现：
```go

```
### 题号：98 验证二叉搜索树
#### 思路：
- 不能单纯比较灭一个节点的左右子节点大小关系
	- 4,3,7,1,8这样的树就是每一个节点的左子节点都 < 它，右子节点都 > 它，但是不是二叉搜索树
- 二叉搜索树的中序遍历是递增序列
- 更好的办法是在递归中用一个值来记录当前最大值maxval，每一次处理的节点都要 > 这个最小值
- 最大值maxval要初始化为可能得最小值
#### c++实现：
```c++
class Solution {
public:
	long long maxVal = LONG_MIN;
    bool isValidBST(TreeNode* root) {
		if (root == nullptr)return true;
		bool lIVB = isValidBST(root->left);
		if (maxVal < root->val){
			maxVal = root->val;
		}else{
			return false;
		}
		bool rIVB = isValidBST(root->right);
		return rIVB && lIVB;
    }
};
```
#### go实现：
```go

```

### 题号：530 二叉搜索树的最小绝对差
#### 思路：
- ~~最小差一定发生在一个节点和它的左子节点或者右子节点之间~~
	- 未必
	- 理解题目要准确
- 最小差一定发生在中序遍历中一个节和它的前一个之间
#### c++实现：
```c++
// 这是相邻节点之间差值最小的算法
class Solution {
public:
	int minDiff = INT_MAX;
    int getMinimumDifference(TreeNode* root) {
        int lmd = minDiff;
        int rmd = minDiff;
		if(root->left){
			lmd = getMinimumDifference(root->left);
		}
		if (root->right){
			rmd = getMinimumDifference(root->right);
		}
		if (root->left && abs(root->val-root->left->val) < minDiff){
			minDiff = abs(root->val-root->left->val);
            cout << minDiff;
		}
		if (root->right && abs(root->val-root->right->val)< minDiff){
			minDiff = abs(root->val-root->right->val);
            cout << minDiff;
		}
        int minv = min(lmd, rmd);
		return min(minDiff, minv);
    }
};

// 这是正确答案
class Solution {
public:
	int minDiff = INT_MAX;
	TreeNode * pre = nullptr;
    int getMinimumDifference(TreeNode* root) {
		if (root == nullptr)return minDiff;
        int lmd = minDiff;
		int rmd = minDiff;
		lmd = getMinimumDifference(root->left);
		if (pre != nullptr){
			minDiff = min(minDiff, root->val-pre->val);
		}
        pre = root;
		rmd = getMinimumDifference(root->right);
		int minv= min(lmd, rmd);
		minDiff = min(minDiff, minv);
		return minDiff;
    }
};
```
#### go实现：
```go

```
### 题号：501 二叉搜索树中的众数
#### 思路：
- 中序遍历：如果前一个节点的值等于当前节点值就是重复元素，重复达到最大重复次数就可以加入众数列表
#### c++实现：
```c++
class Solution {
public:
	int curCount = 0;
	int maxCount = 0;
	TreeNode * pre = nullptr;
	vector<int> res;

	void traversal(TreeNode *cur){
		if (cur == nullptr)return;
		traversal(cur->left);

		if (pre == nullptr){
			curCount = 1;
		}else if (cur->val ==  pre->val){
			curCount++;
		}else{
			curCount=1;
		}

		if (curCount == maxCount){
			res.push_back(cur->val);
		}else if (curCount > maxCount){
			res.clear();
			maxCount = curCount;
			res.push_back(cur->val);
		}
		pre = cur;

		traversal(cur->right);
		return;
	}
    vector<int> findMode(TreeNode* root) {
		traversal(root);
		return res;
    }
};
```
#### go实现：
```go

```
### 题号：236 二叉树的最近公共祖先
#### 思路：
- 递归中搜索一条边的写法大概是if(递归函数(cur->left))retrun;
	- 搜索整棵树：left=递归函数(cur->left)
	- 区别在于返回值是否需要进一步处理
- 遍历整棵树：因为后序遍历要处理完左右子节点后才能返回
- 整体思路
	- 后序遍历
	- 遇到p或者q或者空返回
	- 单层处理：
		- 如果左右子树返回的都不是空，说明当前节点就是要找的节点，返回当前节点
		- 如果一个为空一个不为空，返回不为空的给上一层
#### c++实现：
```c++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
		if(root == p || root == q || root == nullptr)return root;
		TreeNode* left = lowestCommonAncestor(root->left, p, q);
		TreeNode* right = lowestCommonAncestor(root->right, p, q);
		if (left && right){
			return root;
		}else if(!left && right){
			return right;
		}else if(!right && left){
			return left;
		}else{
			return nullptr;
		}
    }
};
```
#### go实现：
```go

```
### 题号：235 二叉搜索树的最近公共祖先
#### 思路：
- 关键在于公共祖先的值v:min(pv, qv)<v<max(p,q) 
- 从上往下的第一个满足v的节点就是最近公共祖先
#### c++实现：
```c++
class Solution {
public:
	TreeNode* res = nullptr;
	void traversal(TreeNode* root, TreeNode* p, TreeNode* q){
		if (root == nullptr)return;
		int minv = min(p->val, q->val);
		int maxv = max(p->val, q->val);
		if (root->val > minv && root->val < maxv){/* 当一个处于p和q之间的值是最大值的时候 */
			if (res == nullptr){
				res = root;
			}
		}else if(root->val == maxv){/* 当p或q中最大值是结果的时候，这时小的值在大的值的左子树 */
			if (res == nullptr){
				res = root;
			}
		}else if (root->val == minv){
			if (res == nullptr){
				res = root;
			}
		}
		lowestCommonAncestor(root->left, p, q);
		lowestCommonAncestor(root->right, p, q);
		return;
	}
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
		traversal(root, p, q);
		return res;
    }
};
```
#### go实现：
```go

```
### 题号：701 二叉搜索树插入
#### 思路：
- 遍历二叉树遇到空位置就插入
- 如果val > 节点值，从节点的右边找空位，反之从左边找
#### c++实现：
```c++
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
		if (root == nullptr){
			TreeNode *node = new(TreeNode);
			node->val = val;
			return node;
		}
		if (root->val > val) root->left = insertIntoBST(root->left,val);
		if(root->val < val)root->right = insertIntoBST(root->right,val);
		return root;
    }
};
```
#### go实现：
```go

```
### 题号：450 删除二叉搜索树中的节点
#### 思路：
- 复杂的点在与找到了要删除的节点dn，并且dn左右子树都存在
	- 删除节点的左孩子成为右子树的最左边节点的左孩子
	- 删除节点的右孩子顶替当前节点
#### c++实现：
```c++
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
		if (root == nullptr)return root;
		if (root->val == key){
			if (!root->left && !root->right){
				delete root;
				return nullptr;
			}
			else if (root->left && !root->right){
				TreeNode *node = root->left;
				delete root;
				return node;
			}
			else if (!root->left && root->right){
				TreeNode *node = root->right;
				delete root;
				return node;
			}
			else if (root->left && root->right){
				TreeNode *leftNode = root->left;
				TreeNode *rightNode = root->right;
				TreeNode *rightLeft = rightNode;
				while(rightLeft->left){
					rightLeft = rightLeft->left;
				}
				rightLeft->left = leftNode;
				delete root;
				return rightNode;
			}
		}
		if (root->val > key) root->left = deleteNode(root->left, key);
		if (root->val < key) root->right =  deleteNode(root->right, key);
		return root;
    }
};
```
#### go实现：
```go

```
### 题号：669 修剪二叉搜索树
#### 思路：
- 如果当前节点值 > 最大值，它的右子树肯定都不在区间内，则递归它的左子树找到在区间内的节点并返回
- 如果当前节点值 < 最小值，递归它的右子树找到区间内的节点返回
- 如果当前节点值在区间内，递归处理它的左右子树，返回的节点作为当前节点的左右孩子
#### c++实现：
```c++
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
		if (root == nullptr)return root;
		if (root->val > high){
			return trimBST(root->left, low, high);
		}else if (root->val < low){
			return trimBST(root->right, low, high);
		}else{
			root->left = trimBST(root->left, low, high);
			root->right = trimBST(root->right, low, high);
			return root;
		}
    }
};
```
#### go实现：
```go

```
### 题号：108 有序数组转二叉搜索树
#### 思路：
- 从数组中间拆分数组，中间元素作为父节点，然后构造左右孩子
#### c++实现：
```c++
class Solution {
public:
	TreeNode *traversal(vector<int>& nums, int left, int right){
		if (left > right)return nullptr;
		int mid = left+(right - left)/2;
		TreeNode *node = new TreeNode(nums[mid]);
		node->left = traversal(nums, left, mid-1);
		node->right = traversal(nums, mid+1, right);
		return node;
	}
    TreeNode* sortedArrayToBST(vector<int>& nums) {
		return traversal(nums, 0, nums.size()-1);
    }
};
```
#### go实现：
```go

```
### 题号：538 二叉搜索树转累加树
#### 思路：
- 右中左顺序遍历，对每一个节点的处理就是加上前一个节点的值 
- 直接在原树上修改值就行，不用重新构建树
#### c++实现：
```c++
class Solution {
public:
	int preVal = 0;
	void traversal(TreeNode* cur){
		if (cur == nullptr)return;
		traversal(cur->right);
		cur->val += preVal;
		preVal = cur->val;
		traversal(cur->left);
	}
    TreeNode* convertBST(TreeNode* root) {
		traversal(root);
		return root;
    }
};
```
#### go实现：
```go

```