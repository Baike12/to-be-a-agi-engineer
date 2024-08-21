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