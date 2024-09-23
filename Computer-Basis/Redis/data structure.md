# 1	redis存储结构
## 1.1	键值对
redis都是以键值对存储的。
- 键：字符串对象
- 值：可以是字符串，也可以是集合类型
redis用hash表保存所有键值对，大概是这么个结构
![[Pasted image 20240920145141.png]]
- key和value就是指向键值**对象**
一个dictEntry：
```c
typedef struct dictEntry {
    //键值对中的键
    void *key;
  
    //键值对中的值
    union {
        void *val;
        uint64_t u64;
        int64_t s64;
        double d;
    } v;
    //指向下一个哈希表节点，形成链表
    struct dictEntry *next;
} dictEntry;
```
- 这里使用union是因为当值就是uint64_t, int64_t, double时直接放到entry中，不需要用一个指针指向
- next指向hash冲突时相同hash值的节点
## 1.2	对象
redis中的对象：
![[Pasted image 20240920145942.png]]
- type：数据类型，比如string，set等
- encoding：数据结构
- ptr：指向底层数据结构的指针
![[Pasted image 20240920173610.png]]
# 2	SDS
redis自己实现了简单动态字符串
## 2.1	c字符串的问题
- 获取字符串长度时间复杂度高
- 以'\ 0'作为结束标识，不能存放二进制数据
- 字符串操作（相加等）不安全不高效，可能造成内存溢出
## 2.2	sds结构
![[Pasted image 20240920152313.png]]
- len：字符数组长度
- alloc：分配的长度，相当于容量
- flags：SDS类型
- buf：底层数据
### 2.2.1	扩容
- 所需空间小于1M：分配所需空间的2倍
- 大于1M：分配所需空间 + 1M
### 2.2.2	SDS类型
有5种SDS类型，每种类型的len和alloc的类型不同，这样就可以根据不同的情况使用不同的类型从而节省内存
![[Pasted image 20240920153547.png]]
还用了__attribute__ ((__packed__))来取消编译对齐

# 3	链表
节点：
```c
typedef struct listNode {
    struct listNode *prev;
    struct listNode *next;
    void *value;
} listNode;
```
list在节点上进行封装
```c
typedef struct list {
    listNode *head;
    listNode *tail;
    void *(*dup)(void *ptr);
    void (*free)(void *ptr);
    int (*match)(void *ptr, void *key);
    unsigned long len;
} list;
```
- 添加了复制、释放、和比对函数还有长度

# 4	压缩列表
## 4.1	结构
![[Pasted image 20240920172313.png]]
- zlbytes：整个压缩列表长度
- zltail：尾部距离头部的偏移量
- zllen：节点数
- zlend：压缩列表结束标识，是一个固定值0xff
## 4.2	entry
![[Pasted image 20240920172548.png]]
- prelen：前一个节点的长度，用于从后往前遍历
	- 如果前一个节点 < 254字节，那么prelen是1字节
	-  >  = 254字节，prelen是5字节
- encoding：记录类型和长度，对于不同类型的数据和不同长度的数据，用不同长度的encoding字段表示
- data：实际数据
## 4.3	连锁更新
由于prelen表示前一个节点和不同长度节点用不同长度prelen表示导致：
- 假设一开始有10个253字节的节点
- 然后插入一个256字节的节点到压缩列表头，这时候第一个253字节的节点会使用5字节的prelen，导致第一个253字节节点自身变成258字节，那么第二个253字节的节点也需要用258字节来表示之前的第一个节点，如此就引发了连锁更新

所以压缩列表不宜太长
# 5	hash表
对于hash冲突使用链式hash
## 5.1	rehash
相同hash的节点太多，影响查询效率，对hash大小进行扩展。
hash表：
```c
struct dict {
    dictType *type;

    dictEntry **ht_table[2];
    unsigned long ht_used[2];

    long rehashidx; /* rehashing not in progress if rehashidx == -1 */

    /* Keep small vars at end for optimal (minimal) struct padding */
    unsigned pauserehash : 15; /* If >0 rehashing is paused */

    unsigned useStoredKeyApi : 1; /* See comment of storedHashFunction above */
    signed char ht_size_exp[2]; /* exponent of size. (size = 1<<exp) */
    int16_t pauseAutoResize;  /* If >0 automatic resizing is disallowed (<0 indicates coding error) */
    void *metadata[];
};
```
这里的dictEntry ** ht_table[2];表示有两个hash表，一般有第一个，在rehash的时候用第二个，流程：
- 给第二个hash表分配两倍于hash表1的内存
- 将表1的数据迁移到表2
- 用表2替代表1，删除原表1，创建空的新表2下次用
**数据量很大的时候，要迁移很多数据**
## 5.2	渐进式hash
主要就是将迁移工作分配到多次：
- redis每次操作时，除了完成命令的操作，还会将涉及的key迁移到表2，直到全部迁移完
- 新增只会新增到表2
## 5.3	触发rehash
负载因子 = hash表的节点个数 / hash表大小，以下条件触发rehash：
- 负载因子 > =1：如果没在RDB快照或者aof重写，就触发
- 负载因子 > =5：强制rehash

# 6	整数集合
```c
typedef struct intset {
    uint32_t encoding;
    uint32_t length;
    int8_t contents[];
} intset;
```
- 实际类型由encoding决定，而不是都是int8_t
## 6.1	整数升级
如果一开始插入的数据都是int16，那整数集合的类型就是int16，然后来了个int32的数据，就会将整数集合的类型升级成int32。
升级的过程是在原数组的位置上增加容量，并保持数据间的顺序不变
可以节省内存。
不支持降级。

# 7	跳表
将节点分层，然后以二分法查找节点
## 7.1	结构
```c
typedef struct zskiplist {
    struct zskiplistNode *header, *tail;
    unsigned long length;
    int level;
} zskiplist;
```
- level：最大层级
## 7.2	节点
```c
typedef struct zskiplistNode {
    sds ele;
    double score;
    struct zskiplistNode *backward;
    struct zskiplistLevel {
        struct zskiplistNode *forward;
        unsigned long span;
    } level[];
} zskiplistNode;
```
实现跳表的关键在level数组，这是一个zskiplistLevel类型的数组，level数组表示一个节点在某一个层的角色，这个角色的意思是这个节点在这一层的前一个节点是谁，和前一个节点之间的跨度是多少
![[Pasted image 20240920225251.png]]
## 7.3	查找节点a流程
从最顶层开始遍历：
- 如果当前节点的score < a的score，遍历当前层的下一个
- 如果当前节点的score = a的score，但是当前节点的ele < a的ele，那么也会遍历当前层的下一个
如果以上两个条件不满足，就遍历到当前节点level数组的下一层
## 7.4	层数设置
对于新创建的节点，随机生成一个数，当这个数 < 0.25，就将新节点的level + 1，直到随机数 > 0.25，返回level、
这样层数越高概率越低
```c
int zslRandomLevel(void) {
    static const int threshold = ZSKIPLIST_P*RAND_MAX;
    int level = 1;
    while (random() < threshold)
        level += 1;
    return (level<ZSKIPLIST_MAXLEVEL) ? level : ZSKIPLIST_MAXLEVEL;
}
```
## 7.5	用跳表的原因
- 跳表在内存占用和查找效率方面和btree差不多，但是跳表实现简单很多
# 8	quicklist
链表加压缩列表，链表的节点是压缩列表
# 9	listpack
用于替换压缩列表，取出了prelen，测地避免了连锁更新
![[Pasted image 20240921012328.png]]