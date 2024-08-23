### 存储层次
#### 寄存器
- 半个时钟周期
#### L1
- 分为数据缓存和指令缓存
- 每个核心都有一个
- 查看命令：数据缓存
- 和寄存器差不多快，2到4个时钟周期
```shell
cat /sys/devices/system/cpu/cpu0/cache/index0/size
32K
```

#### L2
- 每个核心都有
- 10到20个时钟周期
```shell
cat /sys/devices/system/cpu/cpu0/cache/index2/size
1024K
```
#### L3
- 多核心公用
- 20到60时钟周期
```shell
cat /sys/devices/system/cpu/cpu0/cache/index3/size
33792K
```
- 上面都是用的sram
#### 内存
- DRAM：要不断刷新
- 200到300时钟周期

### CPU cache
- cpu要访问一个数据的时候，先到L1cache中去找，没有再到L2cache中找，依次找下去，直到磁盘
- cache由cache line组成，cache line大小
```shell
 cat /sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size
64
```
#### 直接映射
- cpu要读取一个地址数据，地址包括：标签、索引和偏移
	- 索引就是在缓存中的第几行，偏移行中的第几个字节，标签是内存中的地址
- 假设内存数据块64B，缓存1KB，内存32为
	- 缓存有1K / 64B=16行，需要$\log _{2}16=4$ 个位来作索引
	- 需要$\log _{2}64=6$ 位作偏移
	- 剩下的22位就是标签
- 内存地址可以被划分为三部分
	- 先按索引找到缓存行，再比较标签是否匹配，匹配直接按偏移读取，不匹配到内存中去读到缓存

### 提高cache命中率
#### 数据缓存
- 遍历的时候按数据的内存布局顺序遍历可以充分利用缓存
#### 指令缓存
- 使得相同的操作尽量在一起
#### 多核心上提高缓存命中率
- 将计算密集型线程绑定在同一个核心
```c++
int sched_setaffinity(pid_t pid, size_t cpusetsize, const cpu_set_t *mask);
```
### 伪共享
- 多个线程同时读取同一个cache line中的数据导致cache失效的问题。
- 使用__cacheline_aligned避免，将两个变量放在不同cacheline中，对多个线程修改的热点相邻数据使用
```c++
struct{
	int a;
	int b __cacheline_aligned_in_smp;
}
```
### cpu调度
- 对cpu来说进程线程都是task_struct，不过线程里资源共享了进程创建的资源
#### 调度类
- deadline：根据dealine距离当前时间的距离，越近的优先级越高
- realtime：
	- fifo：先到先服务，高优先级可以插队
	- rr：轮流运行，时间片完到队尾，高优先级可以插队
- fair：
	- normal：正常调度策略
	- batch：不和终端交互，可以为其他任务让路
- 完全公平调度：计算虚拟运行时间vruntime
	- 虚拟运行时间越小，优先级越高
	- vruntime=实际运行时间 × nice值 / 权重
#### cpu运行队列
- 每个cpu都有自己的运行队列，其中包含了
	- deadline运行队列
	- realtime运行队列
	- cfs运行队列：一颗红黑树
- dl_rq  >  rt_rq > cfs_rq
- 可以使用nice值调整普通任务优先级（-20到19），越低优先级越高，使用renice调整已经运行中的人物优先级
```shell
nice -n -3 taskname
```
- 使用nice怎么调都是普通任务，可以使用chrt切换为实时任务
```shell
chrt [选项] [PID]
```