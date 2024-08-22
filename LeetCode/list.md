### 题号：203 removeElements
#### 思路：
- 
#### c++实现：
```c++
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
		while(head != nullptr && head->val == val){
			ListNode* tmp = head;
			head = head->next;
			delete tmp;
		}
		ListNode *cur = head;// 此时cur的值肯定不是val，因为最后要返回head，所以用cur来处理
		while(cur != nullptr && cur->next != nullptr){
			if (cur->next->val == val){
				ListNode *tmp = cur->next;
				cur->next = cur->next->next;
				delete tmp;
			}else{
				cur = cur->next;
			}
		}
		return head;
    }
};
```
#### go实现：
```go

```
### 题号：707 disign a list
#### 思路：
- 定义节点：值、next和构造函数
- 定义链表：包含头结点和大小
- 获取指定下标节点
	- 判断下标是否合法
	- 使用while移动到指定节点，返回值
- 在表头添加节点
	- 创建节点，添加到表头，记得递增大小
- 尾部添加
	- 创建节点添加到尾部
- 指定位置添加
	- 先找到要添加的位置的前一个节点nb，添加到nb前面
- 删除指定位置节点
	- 找到要删除的节点的前一个节点nb，删除nb的后一个节点
	- 注意要删除delete内存，并将指针指向null
- 技巧：使用一个虚拟头结点可以方便的找到要处理的节点的前一个节点
- 注意：判断索引的下标的合法性，声明与定义的关系
#### c++实现：
```c++
class MyLinkedList
{
public:
	struct LinkNode
	{
		int val;
		LinkNode *next;
		LinkNode(int val) : val(val), next(nullptr) {}
	};
	MyLinkedList()
	{
		_dummmyHead = new LinkNode(0);
		_size = 0;
	}

	int get(int index)
	{
		if (index > (_size-1) || index < 0){
			return -1;
		}
		LinkNode* cur = _dummmyHead;
		while(index--){
			cur = cur->next;
		}
       
		return cur->next->val;
	}

	void addAtHead(int val)
	{
		LinkNode *newNode = new LinkNode(val);
		newNode->next = _dummmyHead->next;
		_dummmyHead->next = newNode;
		_size++;
      
	}

	void addAtTail(int val)
	{
		LinkNode *newNode = new LinkNode(val);
		LinkNode *cur = _dummmyHead;
		while(cur->next != nullptr){
			cur = cur->next;
		}
		cur->next = newNode;
		_size++;
       
	}

	void addAtIndex(int index, int val)
	{
        
		if(index > _size || index < 0){
			return;
		}
		LinkNode * cur = _dummmyHead;
		while(index--){
			cur = cur->next;
		}
		LinkNode *newNode = new LinkNode(val);
		newNode->next = cur->next;
		cur->next = newNode;
		_size++;
    
	}

	void deleteAtIndex(int index)
	{	
		if(index < 0 || index > (_size-1)){
			return;
		}	
		LinkNode *cur = _dummmyHead;
		while(index--){
			cur=cur->next;
		}
        LinkNode *tmp = cur->next;
		cur->next = cur->next->next;
        delete tmp;
        tmp = nullptr;
		_size--;
   
	}
    void show(){
        LinkNode *cur = _dummmyHead;
        while(cur->next != nullptr)   {
            cout << cur->next->val << "\t";
            cur = cur->next;
        }
        cout << endl;
    }
private:
	int _size;
	LinkNode *_dummmyHead;
};

```
#### go实现：
```go

```
### 题号：206 reverseList
#### 思路：
- 用两个指针cur和pre，一开始cur指向链表第一个节点，pre指向null
- 然后cur的next指向pre，实现一个反转
- 移动pre指向cur
- 移动cur指向cur的next，因为这时cur的next已经被指向了pre，所以需要保存cur本来的next
- 注意：一定要先移动pre再移动cur，如果先移动cur，再移动pre，pre会指向移动后的cur
#### c++实现：
```c++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
		ListNode *cur = head;
		ListNode *pre = nullptr;
		ListNode *originNextOfCur = nullptr;

		while(cur != nullptr){
			originNextOfCur = cur->next;
			cur->next = pre;
            pre = cur;
			cur = originNextOfCur;
		}
		return pre;
    }
};
```
#### go实现：
```go

```
### 题号：24 swapPairs
#### 思路：
- 最好画个图，还有记得删除虚拟头结点
#### c++实现：
```c++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
		ListNode* dummyHead = new ListNode(0);
		dummyHead->next = head;
		ListNode* cur = dummyHead;
		while(cur->next != nullptr && cur->next->next != nullptr){
			ListNode* originNextOfCur = cur->next;
			ListNode* targetNode = cur->next->next->next;
			cur->next = cur->next->next;
			cur->next->next = originNextOfCur;
			originNextOfCur->next = targetNode;

			cur = cur->next->next;
		}

		head = dummyHead->next;
		delete dummyHead;
		dummyHead = nullptr;
		return head;
    }
};
```
#### go实现：
```go

```
### 题号：19 removeNthFromEnd
#### 思路：
- 双指针，快指针先走，有两种先走步数
	- n步：最后快指针指向最后一个节点
	- n+1步，最后快指针指向最后一个节点后一个节点
- 也用虚拟节点，方便处理删除第一个节点的情况
#### c++实现：
```c++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
		ListNode* dummyHead = new ListNode(0);
		dummyHead->next = head;
		ListNode* fast = dummyHead;
		ListNode* slow = dummyHead;
		while(n--){
			fast = fast->next;
		}

		while(fast->next != nullptr){//指向最后一个节点
			fast = fast->next;
			slow = slow->next;
		}
		ListNode* deleteNode = slow->next;
        if(slow->next){
            slow->next = slow->next->next;
        }else{
            slow->next = nullptr;
        }
		
		delete deleteNode;
		deleteNode = nullptr;
		head = dummyHead->next;
		return head;
    }
};
```
#### go实现：
```go

```
### 题号：160 getIntersectionNode无环链表
#### 思路：
- 如果两个链表相交，是尾部对齐的
- 求出两个链表长度差ld，然后长链表先移动ld个位置
- 再一起移动，边移动边判断是否相同
#### c++实现：
```c++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
		int lenA = 0;
		int lenB = 0;
		int diff;
		
		ListNode* count = headA;
		while(count != nullptr){
			lenA ++;
			count = count->next;
		}
		count = headB;
		while(count != nullptr){
			lenB++;
			count = count->next;
		}

		if (lenB > lenA){
			ListNode* tmp = headA;
			headA = headB;
			headB = tmp;
			diff = lenB - lenA;
		}else{
			diff = lenA - lenB;
		}

		while(diff--){
			headA = headA->next;
		}
		
		while(headA != nullptr){
			if(headA == headB){
				return headA;
			}
			headA = headA->next;
			headB = headB->next;
		}
		return nullptr;
    }
};
```
#### go实现：
```go

```
### 题号：142 detectCycle
#### 思路：
- 快慢指针：快指针一次走两步，慢指针一次走一步
- 如果有环，快慢指针一定能相遇，而且第一次相遇时慢指针一定还没走完完整的一圈
- 假设起点到环入口距离为x，第一次相遇点到环口的距离是y，相遇点继续走到环入口的距离是z，有：
	- $2(x+y)=x+y+n(y+z)$ 
	- 当$n=1$ 的时候，有$x=z$ ，就是说这时如果有两个指针，一个从起点开始走，一个从相遇点开始走，两个都是一次只走一步，那么他们的相遇点就是环入口
	- 当$n\neq 1$的时候，相遇点还是环入口点，不过从相遇点出发的指针不是第一次经过环入口
#### c++实现：
```c++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
		ListNode* fast = head;
		ListNode* slow = head;
		while(fast!=nullptr && fast->next != nullptr){
			slow = slow->next;
			fast = fast->next->next;
			if (fast == slow){
				ListNode* p1 = fast;
				ListNode* p2 = head;
				while(p1 != p2){
					p1 = p1->next;
					p2 = p2->next;
				}
				return p1;
			}
		}
		return nullptr;
    }
};
```
#### go实现：
```go

```