### 微调大模型2
#### 预训练：模型底座
#### 微调会导致之前的功能受损
#### 小参数量微调
- 注入参数，只调注入的参数
#### 预训练gpt2
#### karpathy可以低成本预训练
#### olmo模型：整个训练流程
- 卡并行、数据并行
### **关键是注入参数**
#### prompt tuning
- 输入序列前加入伪embeding向量
- 相当于只在输入层加上伪向量
#### p-tuning

#### prefix-tuning
- 在每一层都加上伪向量
### 注入参数矩阵
#### Lora
- 改变QKV矩阵
##### 实现
- 假设要改变Q（d × k）
- 添加一个矩阵A（d × r）和一个矩阵B（r × k）
- 最后的$\mathbf{Q}'=\mathbf{Q}+\mathbf{A}\times \mathbf{B}$ 
##### 效果
- 只注入Q和V效果和注入6个差不多
#### QLora
- 做量化：节省显存
##### 量化

### 实现
#### 数据增强
- chatglm4提供了接口就能做数据增强
#### 在llama3实现function calling

