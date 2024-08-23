### 软中断
- 将中断分为两部分
	- 上半部分用来快速处理终端，比如关闭中断和处理实际敏感的事情，会打断cpu
	- 下半部分处理接下来的任务，用内核线程处理，不会打断cpu，每个cpu都有一个内核线程
- 内核调度也属于软中断
#### 查看中断情况
- 可以看到系统运行以来的中断情况
```shell
cat /proc/softirqs
```
- 查看变化速率
```shell
watch -d cat /proc/softirqs
```
- 软中断都是内核线程处理的，用ps看不到命令
```shell
ps x | grep soft
    6 ?        S      2:13 [ksoftirqd/0]
   14 ?        S      1:56 [ksoftirqd/1]
```
#### 软中断cpu占用高
- 使用top查看是否是软中断导致的cpu占用高
- 使用watch查看不同类型变化率
- 使用sar查看网卡接收速率