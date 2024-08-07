### 训练器
#### 流程
- 定义损失函数
- 定义优化器
- 定义tensorboard的记录器
- 创建用来存放模型的文件夹
- 定义回合数
- 一个回合
	- 训练模式
	- 定义记录损失的列表
	- 使用tqdm显示进度
	- 设置tqdm的进度条
	- 遍历训练集
		- 梯度置0
		- 数据转移到GPU
		- 计算模型数据
		- 计算损失
		- 反向传播
		- 更新网络
		- 步数加1
		- 记录当前损失
			- 将损失从计算图中分离
			- 然后转换成python形式
		- 设置进度条前缀
	- 计算平均损失
	- 验证模式
	- 从验证集中读取特征数据和标签
		- 在不计算梯度的条件下
			- 计算预测值
			- 计算损失
		- 记录损失
	- 计算平均损失
	- 打印损失信息
	- 将训练数据添加到tensorboard
	- 如果平均损失低于最好损失
		- 保存模型参数
		- 打印早停信息
		- 早停计数置0
	- 否则（平均损失大于最好损失）
		- 早停计数加1
	- 如果早停计数大于配置项
		- 停止训练

### 导入数据
#### 流程
- 使用pandas读取出dataframe格式的训练和测试数据
- 提取出numpy类型数据
- 删除dataframe类型节省内存
- 训练数据分割为训练集和验证集
- 特征选择
- 创建数据对象，这里的类继承自pytorch，包括
	- 训练集
	- 验证集
	- 测试集
- 使用DataLoader加载数据集

### 测试
#### 流程
- 保存预测结果
	- 打开一个文件
		- 使用csv将结果写入文件
		- 使用功能
- 使用mymodel类定义模型
- 加载模型参数
- 计算预测值
- 保存预测结果


#### 推理和评估中禁用梯度计算
作用：
- 节约计算
- 节省内存

#### 张量转换成标量
作用：
- 张量有大量元数据，减少内存
- 便于打印和展示
- 便于和其他标量计算
- 避免保留计算图