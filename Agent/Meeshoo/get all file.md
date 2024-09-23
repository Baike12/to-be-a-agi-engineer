### 获取所有文件
1. 第一次访问文件将文件embedding后加入到chroma中，后续要用直接从chroma中获取
	1. 每一个文件就是一个collection
2. 监听文件夹，如果文件有变动就将变动的文件重新embedding
### 问题
- [ ] 如何使用langchain的chroma判断一个collection是否存在
	- 似乎是不行的，因为langchain的chroma创建的时候就是带collection的，如果没有会有默认的collection
	- 所以需要使用chroma客户端来判断