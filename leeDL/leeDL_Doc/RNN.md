### with open
- 打开了一个上下文管理器
![[rnn2.png]]
### 使用fairseq工具将数据二进制化
- 使用fair的进程
	- 指定源语言、目标语言
	- 训练、验证，测试文件前缀
	- 目标路径
	- 使用功能线程数量
### 加载数据
- 使用fairseq的task加载数据集
	- 包括训练集和验证集
	- 对于训练数据集可以使用回译增强数据
	- conbine参数可以让回译数据和原数据合并
		- 回译：
			- 模型A将语言a翻译成语言b
			- 模型B将语言从b翻译成a
#### 数据加载器
- 类似于pytorch中的loader
- 作用
	- 控制一个batch中的token数量
	- 打乱数据集
	- 填充句子到相同长度，实现并行计算
##### 在每个时间步，解码器的输入
- 当前时间步的输入
	- 右移的目标序列
- 右移是为了让解码器看到正确的输出
##### 定义数据加载器
- 参数
	- 任务，数据集类型：训练集或者验证集
- 批量加载器
	- 使用task的获取batch迭代器方法
		- 数据集
		- 最大token和最大句子
		- 最大位置：就是输入序列的最大值
			- 从task中获取
		- 忽略非法输入
		- 种子、线程数
		- 是否使用迭代器缓存：在数据加载时将数据缓存起来，不用每次迭代时都生成批次，设置为True反倒不会生成缓存
##### 加载器流程demo
- 初始化一个加载数据：这玩意儿感觉就是封装了pytorch的加载器
- 初始化一个迭代器：使用

### RNN
- 从fairseq中导入定义好的类
#### 编码器
-原理
- 输入：输入矩阵
- 输出：
	- 输出矩阵
	- 隐状态矩阵
$$\begin{align}
\mathbf{H}_{t} &= \phi(\mathbf{X}_{t}\mathbf{W}_{xh}+\mathbf{H}_{t-1}\mathbf{W}_{hh}+\vec{b}_{h})
\end{align}$$
- 隐状态等于当前实践步的输入加上前一时间步的隐状态，再加上偏置
- 
$$\begin{align}
\mathbf{O}_{t} &= \mathbf{H}_{t}\mathbf{W}_{hq}+\vec{b}_{q}
\end{align}$$
##### 编码器掩码
- 在做注意力时标识哪些位置是填充的掩码

#### RNN编码器实现 
##### 初始化
```python
class RNNEncoder(FairseqEncoder):# 继承fairseq的encoder
    def __init__(self, args, dictionary, embed_tokens):# 词典就是用fairseq生成的，embedtokens就是经过embedding之后的词典
        super().__init__(dictionary)# 使用fairseqencoderc初始化
        self.embed_tokens = embed_tokens
        
        self.num_layers = args.encoder_layers# 隐藏层层数
        
        self.dropout_in_module = nn.Dropout(args.dropout)# dropout系数
        self.rnn = nn.GRU(# 门控循环单元
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=True# 双向
        )
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        self.padding_idx = dictionary.pad()
 
```
**使用门控单元作为模型**
##### 合并正反向隐藏层
```python
    def combine_bidir(self, outs, bsz: int):# 合并正向和反向输出
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()# out形状为：层数（这里如果有多层输出要考虑每一层输出）×双向×批量大小×自动计算出的（序列长度×隐藏层维度）；把批量大小和双向交换位置，方便批量处理数据；最后
        return out.view(self.num_layers, bsz, -1)# 将方向带来的数据和单向输出合并
```
##### 前向传播
```python
    def forward(self, src_tokens, **unused):# rnn前向传播
        bsz, seqlen = src_tokens.size()# 输入序列的批量和序列长度
        
        # 获取 embeddings
        x = self.embed_tokens(src_tokens)# 进行嵌入
        x = self.dropout_in_module(x)# 对嵌入进行暂退

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)# 从批量×序列长度×嵌入维度变成序列长度×批量×嵌入维度，因为rnn期望的是序列长度×批量大小×嵌入维度
        
        # 直通双向 RNN
        h0 = x.new_zeros(2 * self.num_layers, bsz, self.hidden_dim)# 初始化隐藏层
        x, final_hiddens = self.rnn(x, h0)# 将输入和隐藏层传入rnn，得到最终的隐藏层和输出
        outputs = self.dropout_out_module(x)# 输出进行dropout
        # outputs = [sequence len, batch size, hid dim * directions]
        # hidden =  [num_layers * directions, batch size  , hid dim] 
        
        # 由于encode是双向的，我们需要连接两个方向的隐藏状态
        final_hiddens = self.combine_bidir(final_hiddens, bsz)# 合并隐藏层
        # hidden =  [num_layers x batch x num_directions*hidden]
        
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()# 创建掩码：判断srctoken是否和paddingidx相同，paddingidx是一个特定的表示填充的值，相同就表示这个位置是填充的）
        return tuple(
            (
                outputs,  # seq_len x batch x hidden
                final_hiddens,  # num_layers x batch x num_directions*hidden# 每两个层之间
                encoder_padding_mask,  # seq_len x batch为了让解码器知道那些是填充的
            )
        )
```
前向传播整体流程：
- 输入矩阵，获取批量大小和序列维度
- 将矩阵嵌入
- 对嵌入dropout
- 转变形状为rnn需要的：序列长度×批量大小×序列维度
- 初始化隐藏层：2×层数×批量大小×隐藏层维度
- 将隐藏层和嵌入之后的输入输入rnn，得到最后的隐藏层以及输出
- 对输出进行dropout
- 合并隐藏层输出
- 获取掩码层
##### 用于束搜索的函数
```python
    def reorder_encoder_out(self, encoder_out, new_order):
        # 这部分会在fairseq's beam search（束搜索）中使用。
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),# 输出序列
                encoder_out[1].index_select(1, new_order),# 最终隐藏层
                encoder_out[2].index_select(1, new_order),# 编码器填充掩码
            )
        )

```

### 注意力层
```python
class AttentionLayer(nn.Module):
```
#### 初始化
- 定义Q将解码器输出映射到编码器输出的维度
- 定义一个将输出映射的层
```python
def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):# 在编码器的输出和解码器嵌入之后的向量之间计算注意力
        """
        计算decoder embeding 之后 和 encoder out之间的相关程度
        params:
            source_embed_dim: query的维度 $W_Q * I_{decoder-emb}$编码器输出向量的维度
            input_embed_dim: key的维度 $W_K * I_{encoder-out}$解码器嵌入维度
            output_embed_dim: value的维度 $W_V * I_{encoder-out}$
        """
        super().__init__()

        self.Q = nn.Linear(input_embed_dim, source_embed_dim, bias=bias)# 输入维度，隐藏层维度；这里输入是解码器的前一个输出，把解码器的维度转换成编码器输出向量的维度，为了能够点积
        self.output_proj = nn.Linear(#
            input_embed_dim + source_embed_dim, output_embed_dim, bias=bias
        )#
```
#### 定义K、V
- 这里的K和V啥也不干，直接输出
```python
    def K(self, input_tensor):
        return input_tensor
    
    # 这里对encoder-out 不做linear变换
    def V(self, input_tensor):
        return input_tensor
```
#### 前向传播
- 输入：输入形状是解码器时间步×批量大小×特征维度，编码器输出形状是序列长度×批量×特征维度，填充掩码形状是序列长度×批量大小
- 将批量维度放到最前面
- 计算当前输入的Q
- 计算注意力权重
- 填充位置不计算注意力
- 对注意力做softmax
- 和V相乘
```python
    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
    # rnn就喜欢把序列长度放在第一个维度
        # inputs: T, B, dim
        # encoder_outputs: S x B x dim
        # padding mask:  S x B
        
        # 将Batch的维度放在第一
        inputs = inputs.transpose(1,0) # B, T, dim 批量×序列长度×特征维度
        encoder_outputs = encoder_outputs.transpose(1,0) # B, S, dim# 编码器输出作为键和值
        encoder_padding_mask = encoder_padding_mask.transpose(1,0) # B, S
        
        # Q = W_QI_{decode-emb} 投影到编码器输出的维度
        x = self.Q(inputs)

        # 计算 attention
        # (B, T, dim) x (B, dim, S) = (B, T, S)
        # A = K^TQ
        attn_scores = torch.bmm(x, self.K(encoder_outputs.transpose(1,2)))
        # 在与padding相对应的位置取消注意
        if encoder_padding_mask is not None:
            # B, S -> (B, 1, S)
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)
            attn_scores = (
                attn_scores.float()
                .masked_fill_(encoder_padding_mask, float("-inf"))
                .type_as(attn_scores)
            )  # FP16 support: cast to float and back

        # A' = softmax(A)
        attn_scores = F.softmax(attn_scores, dim=-1)

        # O = V A'   (B, T, S) x (B, S, dim) = (B, T, dim) 加权和
        x = torch.bmm(attn_scores, self.V(encoder_outputs))

        # 最终合并 I 和 O  (B, T, dim)
        x = torch.cat((x, inputs), dim=-1)
        x = torch.tanh(self.output_proj(x)) # concat + linear + tanh
        
        # (B, T, dim) -> (T, B, dim)
        return x.transpose(1,0), attn_scores

```
### 解码器
```python
class RNNDecoder(FairseqIncrementalDecoder):
```

#### 初始化
- rnn解码器
- 定义注意力层
- 解码器的输入嵌入和输出投影用相同的参数
	- 简化模型
	- 使用相同的表示空间来读取和生成词，有一致性，也减少了歧义
```python
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.embed_tokens = embed_tokens
        
        assert args.decoder_layers == args.encoder_layers, f"""seq2seq rnn requires that encoder 
        and decoder have same layers of rnn. got: {args.encoder_layers, args.decoder_layers}"""
        assert args.decoder_ffn_embed_dim == args.encoder_ffn_embed_dim*2, f"""seq2seq-rnn requires 
        that decoder hidden to be 2*encoder hidden dim. got: {args.decoder_ffn_embed_dim, args.encoder_ffn_embed_dim*2}"""
        # 编码器层数要等于解码器层数：1、编解码器参数共享时容易实现
        # 解码器隐藏层维度要等于两倍的编码器隐藏层维度：1、解码器不仅要生成输出，也要获取从编码器来的输入，这需要更强的表示能力。2、解码器一般需要将自身隐状态和编码器的隐状态结合
        self.embed_dim = args.decoder_embed_dim# 解码器也有嵌入、隐藏层维度、层数
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers
        
        
        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(# 解码器也是一个rnn
            self.embed_dim, 
            self.hidden_dim, 
            self.num_layers, 
            dropout=args.dropout, 
            batch_first=False, 
            bidirectional=False
        )
        self.attention = AttentionLayer(# 解码器中要定义注意力以从编码器获取数据
            self.embed_dim, self.hidden_dim, self.embed_dim, bias=False
        ) 
        # self.attention = None
        self.dropout_out_module = nn.Dropout(args.dropout)
        
        if self.hidden_dim != self.embed_dim:# 如果隐藏层维度不等于解码器的嵌入维度
            self.project_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)# 需要一个线性层将隐藏层维度投影到输出维度
        else:
            self.project_out_dim = None
        
        if args.share_decoder_input_output_embed:# 输入嵌入和输出投影用相同的参数
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
```
#### 前向传播
- 处理增量解码
```python
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        # 从编码器encoder中提取输出
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out
        # outputs:          seq_len x batch x num_directions*hidden
        # encoder_hiddens:  num_layers x batch x num_directions*encoder_hidden
        # padding_mask:     seq_len x batch
        
        if incremental_state is not None and len(incremental_state) > 0:# 增量解码：模型只关注序列的当前部分，而不是整个序列
            # 如果保留了上一个时间步的信息，我们可以从那里继续，而不是从头开始
            prev_output_tokens = prev_output_tokens[:, -1:]
            cache_state = self.get_incremental_state(incremental_state, "cached_state")
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            # 增量状态不存在，或者这是训练时间，或者是测试时间的第一个时间步
            # 准备 seq2seq: 将encoder_hidden传递给解码器decoder隐藏状态 
            prev_hiddens = encoder_hiddens
        
        bsz, seqlen = prev_output_tokens.size()
        
        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout_in_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
                
        # decoder-to-encoder attention
        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)
                        
        # 直通双向 RNN
        x, final_hiddens = self.rnn(x, prev_hiddens)# 上一步的隐状态
        # outputs = [sequence len, batch size, hid dim]
        # hidden =  [num_layers * directions, batch size  , hid dim]
        x = self.dropout_out_module(x)
                
        # 投影到 embedding size （如果hidden与嵌入embed大小不同，且share_embedding为True，则需要进行额外的投影操作）
        if self.project_out_dim != None:
            x = self.project_out_dim(x)
        
        # 投影到词表大小 vocab size
        x = self.output_projection(x)
        
        # T x B x C -> B x T x C
        x = x.transpose(1, 0)
        
        # 如果是增量，记录当前时间步的隐藏状态，将在下一个时间步中恢复
        cache_state = {
            "prev_hiddens": final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        
        return x, None
```
#### 记录增量状态
```python
    def reorder_incremental_state(
        self,
        incremental_state,
        new_order,
    ):
        # 在fairseq's beam search中使用
        cache_state = self.get_incremental_state(incremental_state, "cached_state")# 获取之前的增量状
        prev_hiddens = cache_state["prev_hiddens"]
        prev_hiddens = [p.index_select(0, new_order) for p in prev_hiddens]# p是一个张量，从张量的第0维按照new_order从新排序，现在变成一个张量列表
        cache_state = {
            "prev_hiddens": torch.stack(prev_hiddens),# 把张量列表变成一个张量，添加第0维
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)# 将从新排序的张量放回去
        return
```
### 序列到序列模型
```python
class Seq2Seq(FairseqEncoderDecoderModel):# 定义序列到序列的模型类
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)# 把自定义的编码器和解码器传入
        self.args = args
    
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
    ):
        """
        前向传播： encoder -> decoder
        """
        encoder_out = self.encoder(# 只需要传入输入序列
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )# 得到编码器输出
        logits, extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return logits, extra
```
### 构建模型
- 参数初始化
```python
def build_model(args, task):
    """基于超参数构建模型实例"""
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

    # token embeddings
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    
    # encoder decoder
    # 提示: TODO: 改用 TransformerEncoder & TransformerDecoder
    encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)
    # encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
    # decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)

    # sequence to sequence model
    model = Seq2Seq(args, encoder, decoder)
    
    # seq2seq 模型初始化很重要, 参数权重的初始化需要一些其他操作
    def init_params(module):
        from fairseq.modules import MultiheadAttention#
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        if isinstance(module, MultiheadAttention):
            module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
            module.v_proj.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.RNNBase):
            for name, param in module.named_parameters():
                if "weight" in name or "bias" in name:
                    param.data.uniform_(-0.1, 0.1)
    # 权重初始化
    model.apply(init_params)
    return model

```
### 优化器
#### 采用平滑损失函数
```python
class LabelSmoothedCrossEntropyCriterion(nn.Module):# 用于替换标准交叉熵损失函数，避免对任何单一标签过于自信
```
- 这里损失函数也是一个线性层
##### 初始化
```python
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing# 平滑系数
        self.ignore_index = ignore_index# 忽略的索引
        self.reduce = reduce# 是否将损失压缩到一个数，是的话对损失求和
```
##### 前向传播
- 计算对数似然损失
- 计算平滑损失：所有对数概率的和取负
```python
        nll_loss = -lprobs.gather(dim=-1, index=target)# 取对数概率中标签位置的值(B)：计算真实值与模型值之间的差
```
- 这一行能计算对数似然损失的原因：
	- 假设模型概率为$\vec{o}=[0.2,0.5,0.3]$ 
	- 真实标签索引为1
	- 那么损失就是-0.5，再取指数，模型概率越接近1，损失值就越小
- 将平滑损失和对数似然损失加权求和得到最终的损失
```python
    def forward(self, lprobs, target):# 对数概率（批量大小B×toekn维度D）、标签批量大小B
        if target.dim() == lprobs.dim() - 1:
            target = target.unsqueeze(-1)# 扩展标签，方便计算
        # nll: 负对数似然（Negative log likelihood），当目标是一个one-ho时的交叉熵。以下行与F.nll_loss相同
        nll_loss = -lprobs.gather(dim=-1, index=target)# 取对数概率中标签位置的值(B)：计算真实值与模型值之间的差
        #  为其他标签保留一些可能性。因此当计算交叉熵时，相当于对所有标签的对数概率求和
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)#（B）基于所有类别的对数概率
        if self.ignore_index is not None:# 去除填充标签的元素
            pad_mask = target.eq(self.ignore_index)
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)# 去除最后一个维度（B
            smooth_loss = smooth_loss.squeeze(-1)
        if self.reduce:
            nll_loss = nll_loss.sum()
            smooth_loss = smooth_loss.sum()
        # 在计算交叉熵时，添加其他标签的损失
        eps_i = self.smoothing / lprobs.size(-1)# 平滑参数除以类别数量D
        loss = (1.0 - self.smoothing) * nll_loss + eps_i * smooth_loss# 负对数似然损失和平滑损失的加权和
        return loss
```
#### 学习率预热
- 逆平方根学习率变化
$$\begin{align}
lr  &= d_{m}^{-0.5}\times min(s ^{-0.5},s\cdot ws ^{-1.5})
\end{align}$$
- 一开始学习率随步数s增加
- 知道s>预热步数
```python
def get_rate(d_model, step_num, warmup_step):# 模型维度
    # TODO: 基于上述公式更新学习率
    # lr = 0.001
    lr = np.power(d_model, -0.5) * min(np.power(step_num, -0.5), step_num * np.power(warmup_step, -1.5))# 维度
    return lr
```
#### 优化器实现
```python
class NoamOpt:
    "实现速率的Optim包装器."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor# 学习率因子
        self.model_size = model_size
        self._rate = 0
    
    @property# 属性方法，获取优化器参数
    def param_groups(self):
        return self.optimizer.param_groups
        
    def multiply_grads(self, c):
        """将梯度乘以常数*c*."""                
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)
        
    def step(self):# 一步优化
        "更新 parameters 和 rate"
        self._step += 1
        rate = self.rate()
        for p in self.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "实现上面的lrate"
        if step is None:
            step = self._step
        return 0 if not step else self.factor * get_rate(self.model_size, step, self.warmup)#
```
### 训练
```python
def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):#
    itr = epoch_itr.next_epoch_itr(shuffle=True)# 数据迭代器
    itr = iterators.GroupedIterator(itr, accum_steps) # 梯度累积：更新每个accum_steps采样，累积accum_stepsh个小批量后才进行更新
    
    stats = {"loss": []}
    scaler = GradScaler() # 自动混合精度`automatic mixed precision` (amp) 根据需要调整梯度大小，防止数值溢出
    
    model.train()
    progress = tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=True)
    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        # 梯度累积：更新每个accum_steps采样
        for i, sample in enumerate(samples):
            if i == 1:
                # 清空CUDA缓存在第一部之后，可以有效减少 OOM 的可能
                torch.cuda.empty_cache()

            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i
            
            # 混合精度训练`mixed precision training`
            with autocast():
                net_output = model.forward(**sample["net_input"])# 计算模型输出
                lprobs = F.log_softmax(net_output[0], -1)# 计算softmax输出的对数：1、取对数可以避免数值溢出；2、在做交叉熵的时候直接用对数softmax计算，减少了交叉熵时的计算；3、对数函数的凸的，方便优化
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))# 计算损失
                
                # logging
                accum_loss += loss.item()
                # 反向传播
                scaler.scale(loss).backward()                
        
        scaler.unscale_(optimizer)# 用到模型更新之前要恢复，之前为了保持数值稳定性放大了
        optimizer.multiply_grads(1 / (sample_size or 1.0)) # (sample_size or 1.0) 处理零梯度的情况
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm) # grad norm 裁剪，处理梯度爆炸
        
        scaler.step(optimizer)
        scaler.update()
        
        # logging
        loss_print = accum_loss/sample_size
        stats["loss"].append(loss_print)
        progress.set_postfix(loss=loss_print)
        if config.use_wandb:
            wandb.log({
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                "train/lr": optimizer.rate(),
                "train/sample_size": sample_size,
            })
        
    loss_print = np.mean(stats["loss"])
    logger.info(f"training loss: {loss_print:.4f}")
    return stats
```
### 模型验证和推理
#### 解码：将模型输出转换张量成句子
```python
sequence_generator = task.build_generator([model], config)# 使用task构建一恶搞序列生成器

def decode(toks, dictionary):
    # 将Tensor装换成我们可阅读的句子(human readable sentence)
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )# 先转移到cpu上
    return s if s else "<unk>"
```
#### 推理步骤
```python
def inference_step(sample, model):#
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []
    for i in range(len(gen_out)):# 验证时，将源序列、模型生成的序列、标签序列都解码成句子后放到列表中，供后续计算blue
        # 对于每个栗子, 收集输入`input`, 翻译结果`hypothesis`和 参考`reference`（label）, 后续用于计算 BLEU
        srcs.append(decode(# 将张量解码为句子后放到列表中
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()), # 对于源序列移除填充
            task.source_dictionary,
        ))
        hyps.append(decode(
            gen_out[i][0]["tokens"], # 0： 表示使用 beam中最靠前的翻译结果# 使用束搜索中最靠前的输出
            task.target_dictionary,
        ))
        refs.append(decode(
            utils.strip_pad(sample["target"][i], task.target_dictionary.pad()), # 移除便签中的填充字符
            task.target_dictionary,
        ))
    return srcs, hyps, refs
```
#### 验证：计算bleu分数
```python
def validate(model, task, criterion, log_to_wandb=True):
    logger.info('begin validation')
    itr = load_data_iterator(task, "valid", 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    stats = {"loss":[], "bleu": 0, "srcs":[], "hyps":[], "refs":[]}
    srcs = []
    hyps = []
    refs = []
    
    model.eval()
    progress = tqdm(itr, desc=f"validation", leave=True)
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # validation loss
            sample = utils.move_to_cuda(sample, device=device)
            net_output = model.forward(**sample["net_input"])

            lprobs = F.log_softmax(net_output[0], -1)
            target = sample["target"]
            sample_size = sample["ntokens"]
            loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
            progress.set_postfix(valid_loss=loss.item())
            stats["loss"].append(loss)
            
            # 模型推理
            s, h, r = inference_step(sample, model)
            srcs.extend(s)
            hyps.extend(h)
            refs.extend(r)
            
    tok = 'zh' if task.cfg.target_lang == 'zh' else '13a'
    stats["loss"] = torch.stack(stats["loss"]).mean().item()
    stats["bleu"] = sacrebleu.corpus_bleu(hyps, [refs], tokenize=tok) # 計算BLEU score
    stats["srcs"] = srcs
    stats["hyps"] = hyps
    stats["refs"] = refs
    
    if config.use_wandb and log_to_wandb:
        wandb.log({
            "valid/loss": stats["loss"],
            "valid/bleu": stats["bleu"].score,
        }, commit=False)
    
    showid = np.random.randint(len(hyps))
    logger.info("example source: " + srcs[showid])
    logger.info("example hypothesis: " + hyps[showid])
    logger.info("example reference: " + refs[showid])
    
    # show bleu results
    logger.info(f"validation loss:\t{stats['loss']:.4f}")
    logger.info(stats["bleu"].format())
    return stats
```
#### 加载模型检查点
```python
def try_load_checkpoint(model, optimizer=None, name=None):
    name = name if name else "checkpoint_last.pt"
    checkpath = Path(config.savedir)/name
    if checkpath.exists():
        check = torch.load(checkpath)
        model.load_state_dict(check["model"])
        stats = check["stats"]
        step = "unknown"
        if optimizer != None:
            optimizer._step = step = check["optim"]["step"]
        logger.info(f"loaded checkpoint {checkpath}: step={step} loss={stats['loss']} bleu={stats['bleu']}")
    else:
        logger.info(f"no checkpoints found at {checkpath}!")
```
#### 定义模型、加载数据集开始训练
```python
model = model.to(device=device)
criterion = criterion.to(device=device)


# In[67]:

# 打印一些日志
logger.info("task: {}".format(task.__class__.__name__))
logger.info("encoder: {}".format(model.encoder.__class__.__name__))
logger.info("decoder: {}".format(model.decoder.__class__.__name__))
logger.info("criterion: {}".format(criterion.__class__.__name__))
logger.info("optimizer: {}".format(optimizer.__class__.__name__))
logger.info(
    "num. model params: {:,} (num. trained: {:,})".format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
)
logger.info(f"max tokens per batch = {config.max_tokens}, accumulate steps = {config.accum_steps}")


# In[68]:

# 加载数据集
epoch_itr = load_data_iterator(task, "train", config.start_epoch, config.max_tokens, config.num_workers)
try_load_checkpoint(model, optimizer, name=config.resume)
while epoch_itr.next_epoch_idx <= config.max_epoch:
    # train for one epoch
    train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
    stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
    logger.info("end of epoch {}".format(epoch_itr.epoch))    
    epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)
```
