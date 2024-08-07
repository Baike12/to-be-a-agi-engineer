#!/usr/bin/env python
# coding: utf-8
from IPython import get_ipython
# #  &#x1F4D1; **作业 5: 机器翻译**
# - 英译中 (繁体)
#   - 输入: an English sentence (比如: tom is a student .)
#   - 输出: the Chinese translation  (比如: 湯姆 是 個 學生 。)
# 
# - <font color=darkred><b>***TODO***:</font></b>
#     - <font color=darkred>训练一个简单的 RNN seq2seq模型，实现翻译功能</font>
#     - <font color=darkred>改用transformer模型提升模型表现</font>
#     - <font color=darkred>Apply Back-translation to furthur boost performance</font>

# # 导入包
# 
# 安装： 
# 
# - **editdistance**： 快速实现编辑距离（Levenshtein距离）。  
# - **sacrebleu**： 计算bleu的库, 可以查看下[知乎: BLEU指标及评测脚本使用的一些误解](https://zhuanlan.zhihu.com/p/404381278)
# - **sacremoses**: 使用Python实现了Moses的tokenizer, truecaser以及normalizer功能，使用起来比较方便[官方Github（有示例）](https://github.com/alvations/sacremoses)
# - **sentencepiece**： 由谷歌将一些词-语言模型相关的论文进行复现，开发了一个开源工具——训练自己领域的sentencepiece模型，该模型可以代替预训练模型(BERT,XLNET)中词表的作用，可以参考[sentencepiece原理与实践](https://zhuanlan.zhihu.com/p/159200073)
# - **wandb**: 是Weights & Biases的缩写，这款工具能够帮助跟踪你的机器学习项目。它能够自动记录模型训练过程中的超参数和输出指标，然后可视化和比较结果，并快速与同事共享结果。[官方文档：quickstart](https://docs.wandb.ai/v/zh-hans/quickstart)
# - **fairseq**: 一个用PyTorch编写的序列建模工具包，它允许研究人员和开发人员训练用于翻译、摘要、语言建模和其他文本生成任务的自定义模型。[fairseq官方文档](https://fairseq.readthedocs.io/en/latest/)

# In[1]:


get_ipython().system('pip install  editdistance  sacrebleu sacremoses sentencepiece wandb')


# In[2]:


# https://fairseq.readthedocs.io/en/latest/
get_ipython().system('git clone https://github.com/pytorch/fairseq.git')
get_ipython().system('cd fairseq && git checkout 9a1c497')
get_ipython().system('pip install --upgrade ./fairseq/')


# In[3]:


import sys
import os
import pdb
import pprint
import logging
import random
import re

import numpy as np
# from tqdm.auto import tqdm 
from tqdm import tqdm # auto下载之后会显示不出来
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from argparse import Namespace
from fairseq import utils
import matplotlib.pyplot as plt


# # 一些功能函数
# 
# 一些重要的方法（随机种子设置）  
# <b>这部分可以不做修改</b>
# 

# In[4]:


def all_seed(seed=6666):
    np.random.seed(seed)
    random.seed(seed)
    # CPU
    torch.manual_seed(seed)
    # GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
    # python全局
    os.environ['PYTHONHASHSEED'] = str(seed)
    # cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'Set env random_seed = {seed}')
    
    
all_seed(73)


# # 数据集下载
# 
# ## En-Zh Bilingual Parallel Corpus  （英-中 平行语料）
# * [TED2020](#reimers-2020-multilingual-sentence-bert)
#     - Raw: 398,066 (sentences)   
#     - Processed: 393,980 (sentences)
# 
# ## 测试数据
# - 大小: 4,000 (句子)
# - **中文翻译并未提供。 提供的（.zh）文件是伪翻译，每一行都是一个'。'**

# In[5]:


data_dir = './DATA/rawdata'
dataset_name = 'ted2020'
urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted2020.tgz",
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/test.tgz",
)
file_names = (
    'ted2020.tgz', # train & dev
    'test.tgz', # test
)
prefix = Path(data_dir).absolute() / dataset_name
# 下载数据
prefix.mkdir(parents=True, exist_ok=True)
for u, f in zip(urls, file_names):
    path = prefix/f
    # 不存在则直接通过 weget进行下载
    if not path.exists():
        get_ipython().system('wget {u} -O {path}')
    if path.suffix == ".tgz":
        get_ipython().system('tar -xvf {path} -C {prefix}')
    elif path.suffix == ".zip":
        get_ipython().system('unzip -o {path} -d {prefix}')
get_ipython().system("mv {prefix/'raw.en'} {prefix/'train_dev.raw.en'}")
get_ipython().system("mv {prefix/'raw.zh'} {prefix/'train_dev.raw.zh'}")
get_ipython().system("mv {prefix/'test/test.en'} {prefix/'test.raw.en'}")
get_ipython().system("mv {prefix/'test/test.zh'} {prefix/'test.raw.zh'}")
get_ipython().system("rm -rf {prefix/'test'}")


# ## 文件数据查看

# In[6]:


src_lang = 'en'
tgt_lang = 'zh'

data_prefix = f'{prefix}/train_dev.raw'
test_prefix = f'{prefix}/test.raw'


# In[7]:


# 查看数据
get_ipython().system("head {data_prefix+'.'+src_lang} -n 5")
get_ipython().system("head {data_prefix+'.'+tgt_lang} -n 5")


# # 数据处理
# 
# 
# 1. 全角转半角 `strQ2B`
# 2. 一些字符串替换 `clean_s`
# 3. 训练数据适当的清洗 `clean_corpus`
#     - 删除过短数据 `min_len`
#     - 删除过长数据 `max_len`
#     - 删除翻译前后数据长度比例超过一定值的字段 `ratio`

# In[8]:


def strQ2B(ustring):
    """
    把字符串全角转半角
    """
    # 参考: https://cloud.tencent.com/developer/article/1435475
    ss = []# 创建一个列表
    for s in ustring:# 遍历输入字符串
        rstring = ""
        for uchar in s:# 遍历字符
            inside_code = ord(uchar)# 获取编码
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):   # 全角字符（除空格）根据关系转化，>65281并且<65374
                inside_code -= 65248# 转换
            rstring += chr(inside_code)# 转换回字符
        ss.append(rstring)# 添加到返回的字符串
    return ''.join(ss)# 返回字符串
# function:全角转半角

def clean_s(s, lang):
    if lang == 'en':
        s = re.sub(r"\([^()]*\)", "", s) # 删除 ([text])
        s = s.replace('-', '') # 删除 '-'
        s = re.sub('([.,;!?()\"])', r' \1 ', s) # 保留标点符号
    elif lang == 'zh':
        s = strQ2B(s) # 把字符串全角转半角
        s = re.sub(r"\([^()]*\)", "", s) # 删除 ([text])
        s = s.replace(' ', '')
        s = s.replace('—', '')
        s = s.replace('“', '"')
        s = s.replace('”', '"')
        s = s.replace('_', '')
        s = re.sub('([。,;!?()\"~「」])', r' \1 ', s) # 保留标点符号
    s = ' '.join(s.strip().split())# 删掉首尾空格，切分成列表，在列表元素之间加上空格
    return s
# function:清理数据


def len_s(s, lang):
    if lang == 'zh':
        return len(s)# 中文返回s长度，也就是汉字个数
    return len(s.split())# 英文返回单词个数


def clean_corpus(prefix, l1, l2, ratio=9, max_len=1000, min_len=1):# l1是en或者zh
    if Path(f'{prefix}.clean.{l1}').exists() and Path(f'{prefix}.clean.{l2}').exists():
        print(f'{prefix}.clean.{l1} & {l2} exists. skipping clean.')
        return# 如果文件存在了，直接返回，不处理了
    with open(f'{prefix}.{l1}', 'r') as l1_in_f:
        with open(f'{prefix}.{l2}', 'r') as l2_in_f:# 打开
            with open(f'{prefix}.clean.{l1}', 'w') as l1_out_f:# 写入
                with open(f'{prefix}.clean.{l2}', 'w') as l2_out_f:
                    for s1 in tqdm(l1_in_f):# 遍历文件1
                        s1 = s1.strip()# 切分文件1
                        s2 = l2_in_f.readline().strip()# 切分文件2
                        s1 = clean_s(s1, l1)# 清理
                        s2 = clean_s(s2, l2)
                        s1_len = len_s(s1, l1)
                        s2_len = len_s(s2, l2)
                        if min_len > 0: # 删除过短数据
                            if s1_len < min_len or s2_len < min_len:
                                continue
                        if max_len > 0: # 删除过长数据
                            if s1_len > max_len or s2_len > max_len:
                                continue
                        if ratio > 0: # 删除翻译前后数据长度比例超过一定值的字段
                            if s1_len/s2_len > ratio or s2_len/s1_len > ratio:
                                continue
                        print(s1, file=l1_out_f)
                        print(s2, file=l2_out_f)


# In[9]:


get_ipython().system('rm -f /kaggle/working/DATA/rawdata/ted2020/train_dev.raw.clean.en')
get_ipython().system('rm -f /kaggle/working/DATA/rawdata/ted2020/train_dev.raw.clean.zh')


# In[10]:


clean_corpus(data_prefix, src_lang, tgt_lang)
clean_corpus(test_prefix, src_lang, tgt_lang, ratio=-1, min_len=-1, max_len=-1)


# In[11]:


# view data
get_ipython().system("head {data_prefix+'.clean.'+src_lang} -n 5")
get_ipython().system("head {data_prefix+'.clean.'+tgt_lang} -n 5")


# # 数据拆分

# In[12]:


valid_ratio = 0.01 # 3000~4000 就足够了
train_ratio = 1 - valid_ratio


# In[13]:


if (prefix/f'train.clean.{src_lang}').exists() \
and (prefix/f'train.clean.{tgt_lang}').exists() \
and (prefix/f'valid.clean.{src_lang}').exists() \
and (prefix/f'valid.clean.{tgt_lang}').exists():
    print(f'train/valid splits exists. skipping split.')# 这次处理之后的文件名类似train.clean.zh如果文件存在就打印并跳过
else:
    line_num = sum(1 for line in open(f'{data_prefix}.clean.{src_lang}'))# 计算未拆分的文件的行数
    labels = list(range(line_num))# 初始化标签
    random.shuffle(labels)# 打乱标签
    for lang in [src_lang, tgt_lang]:
        train_f = open(os.path.join(data_dir, dataset_name, f'train.clean.{lang}'), 'w')# 新建训练集文件
        valid_f = open(os.path.join(data_dir, dataset_name, f'valid.clean.{lang}'), 'w')# 新建验证集文件
        count = 0
        # 基于下标拆分训练和测试集
        for line in open(f'{data_prefix}.clean.{lang}', 'r'):# 打开拆分前的文件
            if labels[count]/line_num < train_ratio:# 如果label中第count个元素除以行数<训练集比例，labels是一个随机的个数等于行数的列表，相当于随机取训练集个数个行写到训练集文件中
                train_f.write(line)
            else:
                valid_f.write(line)
            count += 1
        train_f.close()
        valid_f.close()


# ## &#x2728; 子字单元(`Subword Units`) 
# 在机器翻译中词表外词（OOV）的翻译是主要问题。 这个问题可以通过使用子字单元(`Subword Units`) 来缓解。
# - 我们使用 [sentencepiece](#kudo-richardson-2018-sentencepiece) python包
# - 选择“unigram”或“字节对编码（BPE）”算法

# In[14]:


import sentencepiece as spm

vocab_size = 8000# 词表长度
if (prefix/f'spm{vocab_size}.model').exists():# 模型在就不训了
    print(f'{prefix}/spm{vocab_size}.model exists. skipping spm_train.')
else:
    spm.SentencePieceTrainer.train(# 训练一个分词器，使得模型更容易接受
        input=','.join([f'{prefix}/train.clean.{src_lang}',# 训练集和验证集的文件列表
                        f'{prefix}/valid.clean.{src_lang}',
                        f'{prefix}/train.clean.{tgt_lang}',
                        f'{prefix}/valid.clean.{tgt_lang}']),
        model_prefix=prefix/f'spm{vocab_size}',# 训练的模型存放地
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 用'bpe'也行 
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )


# In[15]:


# 用训练好的模型清洗数据: 将句子的起始终点加上标识：
# 如 ▁這個 研 討 會 給我 留 下 了 極 為 深 刻 的 印 象 ▁, ▁我想 感 謝 大家 對我 之前 演講 的 好 評 
spm_model = spm.SentencePieceProcessor(model_file=str(prefix/f'spm{vocab_size}.model'))# 加载模型
in_tag = {
    'train': 'train.clean',
    'valid': 'valid.clean',
    'test': 'test.raw.clean',
}
for split in ['train', 'valid', 'test']:# 对训练集、验证集、测试集分词
    for lang in [src_lang, tgt_lang]:
        out_path = prefix/f'{split}.{lang}'# 组成文件名train.en这是最终的名字
        if out_path.exists():
            print(f"{out_path} exists. skipping spm_encode.")
        else:
            with open(prefix/f'{split}.{lang}', 'w') as out_f:# 打开最终文件
                with open(prefix/f'{in_tag[split]}.{lang}', 'r') as in_f:# 打开读取的文件
                    for line in in_f:# 遍历行
                        line = line.strip()# 将行拆分
                        tok = spm_model.encode(line, out_type=str)# 使用训练好的模型编码当前行
                        print(' '.join(tok), file=out_f)# 将编码后的token输出到文件中


# In[16]:


get_ipython().system('head -n 10 /kaggle/working/DATA/rawdata/ted2020/train.zh')


# ## &#x2728; 使用fairseq将数据二进制化
# 
# 生成4个文件
# - train.en-zh.en.bin
# - train.en-zh.en.idx
# - train.en-zh.zh.bin
# - train.en-zh.zh.idx

# In[17]:


binpath = Path('./DATA/data-bin', dataset_name)
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    get_ipython().system("python -m fairseq_cli.preprocess          --source-lang {src_lang}         --target-lang {tgt_lang}         --trainpref {prefix/'train'}         --validpref {prefix/'valid'}         --testpref {prefix/'test'}         --destdir {binpath}         --joined-dictionary         --workers 2")


# In[18]:


print(binpath)
get_ipython().system('ls DATA/data-bin/ted2020 | sort')
get_ipython().system('head -n 2 DATA/data-bin/ted2020/train.en-zh.en.idx')


# # 超参设置
# `config` 包含所有训练需要的超参数（便于后续的调参），以及模型需要存储的位置

# In[19]:


config = Namespace(# 超参数
    datadir = "./DATA/data-bin/ted2020",
    savedir = "./checkpoints/rnn",
    source_lang = "en",
    target_lang = "zh",
    
    # 设置cpu核数：fetching & processing data.
    num_workers=2,  
    # batch size 中tokens大小设置. 梯度累积增加有效批量 gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,
    
    # NoamScheduler调整学习率。我们可以通过 lr_factor 调整最大lr。
    #     NoamScheduler：可以在numpy-ml中了解详情 https://numpy-ml.readthedocs.io/en/latest/numpy_ml.neural_nets.schedulers.html
    #         lr= lr_factor * ( model_size ** (-0.5) * min(step** (-0.5), step * warmup_steps ** (-1.5)) )
    lr_factor=2.,
    lr_warmup=4000,
    
    # 梯度裁剪norm ，防止梯度爆炸
    clip_norm=1.0,
    
    # 训练最大轮次
    max_epoch=15,
    start_epoch=1,
    
    # beam search 大小
    #    beam search 可以详细阅读《动手学深度学习》： https://d2l.ai/chapter_recurrent-modern/beam-search.html
    beam=5, 
    # 生成序列最大长度 ax + b, x是原始序列长度
    max_len_a=1.2, 
    max_len_b=10, 
    # 解码时，数据后处理：删除句子符号 和 jieba对句子 。
    # when decoding, post process sentence by removing sentencepiece symbols and jieba tokenization.
    post_process = "sentencepiece",
    
    # checkpoints
    keep_last_epochs=5,
    resume=None, # 如果设置，则从config.savedir 中的对应 checkpoint name 恢复
    
    # logging
    use_wandb=False,
    
    # 随机种子seed
    seed=73
)


# # 日志

# In[20]:


logging.basicConfig(# 日志格式
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",# 形式
    datefmt="%Y-%m-%d %H:%M:%S",# 时间格式
    level="INFO", # "DEBUG" "WARNING" "ERROR"# 日志级别
    stream=sys.stdout,# 输出到何处
)
# function:定义日志的格式、时间格式、日志级别、输出到何处
proj = "hw5.seq2seq"# 项目名称
logger = logging.getLogger(proj)# 创建日志记录器
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)


# In[21]:


cuda_env = utils.CudaEnvironment()# 设置cuda环境
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')# 设置计算设备


# # 数据集加载
# ## 我们从fairseq借用TranslationTask
# * 用于加载上面创建的二进制数据
# * 实现数据迭代器 (dataloader)
# * 内置 task.source_dictionary 和 task.target_dictionary 同样方便
# * 实现 beach search decoder

# In[22]:


from fairseq.tasks.translation import TranslationConfig, TranslationTask# 导入翻译配置和翻译认为模块，这个模块就是一个现成的翻译模型

task_cfg = TranslationConfig(# 翻译配置
    data=config.datadir,# 数据路径
    source_lang=config.source_lang,# 原语言
    target_lang=config.target_lang,# 目标语言
    train_subset="train",# 训练集名称
    required_seq_len_multiple=8,#
    dataset_impl="mmap",# 加载数据集的方式，使用mmap映射到进程内存
    upsample_primary=1,
)
task = TranslationTask.setup_task(task_cfg)


# In[23]:


logger.info("loading data for epoch 1")# 打印一行日志
task.load_dataset(split="train", epoch=1, combine=True) # combine if you have back-translation data.加载训练数据
task.load_dataset(split="valid", epoch=1)# 加载验证数据集


# In[24]:


sample = task.dataset("valid")[1]
pprint.pprint(sample)
pprint.pprint(
    "Source: " + \
    task.source_dictionary.string(
        sample['source'],
        config.post_process,
    )
)
pprint.pprint(
    "Target: " + \
    task.target_dictionary.string(
        sample['target'],
        config.post_process,
    )
)


# # 数据生成器
# > 和 pytorch 中的 `torch.utils.data.DataLoader`类似
# 
# * 控制每个`batch`包含不超过N个`token`，从而优化GPU内存效率
# * 每个`batch`对训练集进行重排序（乱序）
# * 忽略超过最大长度的句子
# * 将`batch`中的所有句子填充到相同的长度，从而实现GPU的并行计算
# * 在一个`token`中增加 `eos` 和 `shift`
#     - **训练的时候注意**: 为了训练模型以基于前缀预测下一个词语，我们将右移的目标序列作为decoder输入。
#     - 一般来说，将bos前置到目标就可以完成任务（如下所示）
# ![seq2seq](https://i.imgur.com/0zeDyuI.png)
#     - 在`fairseq`中进行的处理有点不样,是将`eos`移动到最开始位置。根据经验，这也有同样的效果。示例:
#     ```
#     # output target (target) and Decoder input (prev_output_tokens): 
#                    eos = 2
#                 target = 419,  711,  238,  888,  792,   60,  968,    8,    2
#     prev_output_tokens = 2,  419,  711,  238,  888,  792,   60,  968,    8
#     ```

# In[25]:


seed=config.seed

def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):# 数据加载器
    batch_iterator = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=max_tokens,
        max_sentences=None,
        # fairseq.utils
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            max_tokens,
        ),
        ignore_invalid_inputs=True,
        seed=seed,
        num_workers=num_workers,
        epoch=epoch,
        disable_iterator_cache=not cached,
        # 将此设置为False以加快速度。但是，如果设置为False，则将max_tokens更改为
        # 此方法的第一次调用无效. 
    )
    return batch_iterator

demo_epoch_obj = load_data_iterator(task, "valid", epoch=1, max_tokens=20, num_workers=1, cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
sample = next(demo_iter)
sample


# * 每个batch是python dict, key是`string`, values是`Tensor value`.
# ```python
# batch = {
#     "id": id, # 样例的id
#     "nsentences": len(samples), # 句子批次大小 (sentences)
#     "ntokens": ntokens, # token批次大小 (tokens)
#     "net_input": {
#         "src_tokens": src_tokens, # 翻译的句子(`source language`)
#         "src_lengths": src_lengths, # 句子padding前的长度
#         "prev_output_tokens": prev_output_tokens, # 右移动的目标（`right shifted target`)
#     },
#     "target": target, # 翻译结果
# }
# ```

# #   &#x2728; 模型

# In[26]:


from fairseq.models import (
    FairseqEncoder, 
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel
)# 从fairseq中导入


# ## Encoder

# - The Encoder 可以是 `RNN`  或 `Transformer Encoder`. 下面的描述是针对RNN的. 对于每个输入 token, Encoder 会生成一个 output vector 和一个 hidden states vector, 并且 hidden states vector 将传入下一个rnn cell. 换句话来说, Encoder 按输入顺序顺序读取句子, 并在每个时间步输出一个 single vector, 最终输出final hidden states, 或者 content vector, 在最后一个时间步.
# - 超参数:
#   - *args*
#       - encoder_embed_dim: embeddings维度, 将 one-hot 向量压缩, 实现维度降低
#       - encoder_ffn_embed_dim：  hidden states 和 output vectors 的维度
#       - encoder_layers：RNN Encoder的层数
#       - dropout：将部分神经元激活的概率被设置为0，以防止过度拟合。通常，这在训练中应用，在测试中删除。
#   - *dictionary*: fairseq 中提供的dictionary.它用于获得填充索引(`padding index`)，进而获得编码器填充掩码(`encoder padding mask`)。
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# 
# - 输入: 
#     - *src_tokens*: 表示英语的整数序列 e.g. 1, 28, 29, 205, 2 
# - 输出: 
#     - *outputs*:  RNN每步输出,可以进行Attention处理
#     - *final_hiddens*: 每一步的`hidden states`, 传入解码器(`decoder`)用于翻译
#     - *encoder_padding_mask*: 这告诉解码器(`decoder`)忽略哪个位置

# In[27]:


class RNNEncoder(FairseqEncoder):# 继承fairseq的encoder
    def __init__(self, args, dictionary, embed_tokens):# 词典就是用fairseq生成的，embedtokens就是经过embedding之后的词典
        super().__init__(dictionary)# 使用fairseqencoderc初始化
        self.embed_tokens = embed_tokens
        
        self.embed_dim = args.encoder_embed_dim# 嵌入维度
        self.hidden_dim = args.encoder_ffn_embed_dim# 隐藏层维度
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
        # function:继承fairseq定义了encoder层
        
    def combine_bidir(self, outs, bsz: int):# 合并正向和反向输出
        out = outs.view(self.num_layers, 2, bsz, -1).transpose(1, 2).contiguous()# out形状为：层数（这里如果有多层输出要考虑每一层输出）×双向×批量大小×自动计算出的（序列长度×隐藏层维度）；把批量大小和双向交换位置，方便批量处理数据；最后
        return out.view(self.num_layers, bsz, -1)# 将方向带来的数据和单向输出合并

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
                encoder_padding_mask,  # seq_len x batch
            )
        )
    
    def reorder_encoder_out(self, encoder_out, new_order):
        # 这部分会在fairseq's beam search（束搜索）中使用。
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),# 输出序列
                encoder_out[1].index_select(1, new_order),# 最终隐藏层
                encoder_out[2].index_select(1, new_order),# 编码器填充掩码
            )
        )


# ## Attention
# - 当输入序列较长时，“内容向量”单独不能准确地表示整个序列，注意力机制(`Attention`)可以为解码器(`Encoder`)提供更多信息。
# - 根据当前时间步长的**Decoder embeddings**，将**Encoder outputs**与解码器嵌入(`Decoder embeddings`)匹配，以确定相关性，然后将编码器输出与相关性加权相加，作为**Decoder**RNN的输入。 
# - 常见的注意力实现使用神经网络/点积作为**query**（解码器嵌入`decoder embeddings`）和**key**（编码器输出`Encoder outputs`）之间的相关性，然后是**softmax**以获得分布，最后**values**（编码器输入`Encoder outputs`）被所述分布**加权和**。
#     - $Q = W_Q  I_{decoder-emb}$
#     - $K = W_K  I_{encoder-out}$
#     - $V = W_V  I_{encoder-out}$
#     - $A = K^TQ$
#     - $A'= softmax(A)$
#     - $O = VA'$
# 
# - Parameters:
#   - *source_embed_dim*: query的维度 $W_Q$
#   - *input_embed_dim*: key的维度 $W_K$
#   - *output_embed_dim*: value的维度 $W_V$
# 
# - Inputs: 
#     - *inputs*: 做attention的输入
#     - *encoder_outputs*:  作为 query / value,
#     - *encoder_padding_mask*: 这告诉解码器`decoder`忽略哪个位置
# - Outputs: 
#     - *output*: attention后的上下文向量
#     - *attention score*: attention后的分布 $A'$

# In[28]:


class AttentionLayer(nn.Module):
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
    
    # 这里对encoder-out 不做linear变换
    def K(self, input_tensor):
        return input_tensor
    
    # 这里对encoder-out 不做linear变换
    def V(self, input_tensor):
        return input_tensor

    def forward(self, inputs, encoder_outputs, encoder_padding_mask):
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
        x = torch.cat((x, inputs), dim=-1)#
        x = torch.tanh(self.output_proj(x)) # concat + linear + tanh
        
        # (B, T, dim) -> (T, B, dim)
        return x.transpose(1,0), attn_scores


# ## Decoder
# 
# * **Decoder**的隐藏状态(`hidden states`)将由**Encoder**的最终隐藏状态（`final hidden states`）初始化
# * 同时，**Decoder**将根据当前时间步长的输入（之前时间步长的输出）更改其隐藏状态（`hidden states`），并生成输出
# * Attention提高了性能
# * seq2seq步骤在解码器(`decoder`)中实现 , 以便以后Seq2Seq类可以接受RNN和Transformer，而无需进一步修改。
# 
# - Parameters:
#   - *args*
#       - decoder_embed_dim: decoder 维度, 和`encoder_embed_dim`类似，
#       - decoder_ffn_embed_dim: decoder RNN 的隐含层(hidden states`)维度,和`encoder_ffn_embed_dim`类似
#       - decoder_layers: decoder RNN 的网络层数
#       - share_decoder_input_output_embed: 通常，解码器`decoder`的投影矩阵将与解码器输入嵌入(`decoder input embeddings`)共享权重
#   - *dictionary*: fairseq 中提供的dictionary.
#   - *embed_tokens*: an instance of token embeddings (nn.Embedding)
# 
# - Inputs: 
#     - *prev_output_tokens*: 表示右移目标(`right-shifted target`)的整数序列  e.g. 1, 28, 29, 205, 2 
#     - *encoder_out*: encoder的输出.
#     - *incremental_state*: 为了在测试期间加快解码速度，我们将保存每个时间步长的隐藏状态(` hidden state`)。
# 
# - Outputs: 
#     - *outputs*: 解码器每个时间步的logits（softmax之前）输出
#     - *extra*: 没有使用

# In[29]:


class RNNDecoder(FairseqIncrementalDecoder):
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


# ## Seq2seq

# In[30]:


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


# ## &#x2728; 模型初始化
# 
# <font color=darkred><b>***TODO***: Encoder和Decoder修改：改用 TransformerEncoder & TransformerDecoder</font></b>

# In[31]:


# 提示: transformer 结构
from fairseq.models.transformer import (
    TransformerEncoder, 
    TransformerDecoder,
)

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


# ## 模型框架一些参数配置
# 
# 对于强基线，请参考表3中*transformer-base*的超参数 [Attention is all you need](#vaswani2017)

# In[32]:


arch_args = Namespace(
    encoder_embed_dim=256,
    encoder_ffn_embed_dim=512,
    encoder_layers=1,
    decoder_embed_dim=256,
    decoder_ffn_embed_dim=1024,
    decoder_layers=1,
    share_decoder_input_output_embed=True,
    dropout=0.3,
)

# 提示: 这些是关于Transformer参数的补丁 
def add_transformer_args(args):
    args.encoder_attention_heads=4
    args.encoder_normalize_before=True
    
    args.decoder_attention_heads=4
    args.decoder_normalize_before=True
    
    args.activation_fn="relu"
    args.max_source_positions=1024
    args.max_target_positions=1024
    
    # Transformer默认参数上的修补程序 (以上未列出)
    from fairseq.models.transformer import base_architecture
    base_architecture(arch_args)

# add_transformer_args(arch_args)


# In[33]:


if config.use_wandb:
    wandb.config.update(vars(arch_args))

    
model = build_model(arch_args, task)
logger.info(model)


# # 优化器
# 
# ## 损失函数: 标签平滑正则化（`Label Smoothing Regularization`)
# * 让模型学习生成不太集中的分布，并防止过度自信(`over-confidence`)
# * 有时，事实真相可能不是唯一的答案。因此，在计算损失时，我们为不正确的标签保留一些概率
# * 防止过拟合
# 
# code [来源：fairseq/criterions/label_smoothed_cross_entropy](https://fairseq.readthedocs.io/en/latest/_modules/fairseq/criterions/label_smoothed_cross_entropy.html)

# In[34]:


class LabelSmoothedCrossEntropyCriterion(nn.Module):# 用于替换标准交叉熵损失函数，避免对任何单一标签过于自信
    def __init__(self, smoothing, ignore_index=None, reduce=True):
        super().__init__()
        self.smoothing = smoothing# 平滑系数
        self.ignore_index = ignore_index# 忽略的索引
        self.reduce = reduce# 是否将损失压缩到一个数，是的话对损失求和
    
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

# 通常来说, 0.1 就足够了
criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_index=task.target_dictionary.pad(),
)


# ## &#x2728; 优化器: Adam + lr scheduling
# 在训练Transformer时，逆平方根学习率变化对稳定性很重要. 它后来也用于RNN。
# 根据以下公式更新学习率`learning rate` . 在第一阶段，线性增加学习率，然后学习率与时间步长的平方根倒数成比例衰减。
# $$lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$$

# In[45]:


def get_rate(d_model, step_num, warmup_step):# 模型维度
    # TODO: 基于上述公式更新学习率
    # lr = 0.001
    lr = np.power(d_model, -0.5) * min(np.power(step_num, -0.5), step_num * np.power(warmup_step, -1.5))# 维度
    return lr


# In[46]:


# 可以看：https://nn.labml.ai/optimizers/noam.html
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


# ## 学习率Scheduling图示

# In[50]:


optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim, 
    factor=config.lr_factor, 
    warmup=config.lr_warmup, 
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

plt.title("lr Scheduling:\n $lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$")
plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
None


# #   &#x2728; 训练部分

# In[60]:


from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

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
        
        scaler.unscale_(optimizer)
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


# ## 模型验证 & 模型推理(预测)
# 为了防止过度拟合，每个`epoch`都需要进行验证，以验证对未见数据的性能
# - 该程序本质上与训练相同，就是增加了推理（预测）步骤
# - 验证后，我们可以保存模型权重
# 
# 仅验证损失无法描述模型的实际性能
# - 基于当前模型直接生成翻译结果，然后使用参考译文（目标值）计算BLEU
# - 我们还可以手动检查翻译结果的质量
# - 我们使用fairseq序列生成器进行`beam search`以生成多个翻译结果

# In[53]:


# fairseq's beam search 生成器
# 基于模型和给定输入序列, 通过beam search生成翻译结果
sequence_generator = task.build_generator([model], config)# 使用task构建一恶搞序列生成器

def decode(toks, dictionary):
    # 将Tensor装换成我们可阅读的句子(human readable sentence)
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )# 先转移到cpu上
    return s if s else "<unk>"

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


# In[54]:


import shutil
import sacrebleu

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


# # 保存和载入模型权重

# In[65]:


def validate_and_save(model, task, criterion, optimizer, epoch, save=True):   
    stats = validate(model, task, criterion)
    bleu = stats['bleu']
    loss = stats['loss']
    if save:
        # 保存 epoch checkpoints
        savedir = Path(config.savedir).absolute()
        savedir.mkdir(parents=True, exist_ok=True)
        
        check = {
            "model": model.state_dict(),
            "stats": {"bleu": bleu.score, "loss": loss},
            "optim": {"step": optimizer._step}
        }
        torch.save(check, savedir/f"checkpoint{epoch}.pt")
        shutil.copy(savedir/f"checkpoint{epoch}.pt", savedir/f"checkpoint_last.pt")
        logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")
    
        # 保存 epoch 例子
        with open(savedir/f"samples{epoch}.{config.source_lang}-{config.target_lang}.txt", "w") as f:
            for s, h in zip(stats["srcs"], stats["hyps"]):
                f.write(f"{s}\t{h}\n")

        # 获取验证中最佳的 bleu    
        if getattr(validate_and_save, "best_bleu", 0) < bleu.score:
            validate_and_save.best_bleu = bleu.score
            torch.save(check, savedir/f"checkpoint_best.pt")
            
        del_file = savedir / f"checkpoint{epoch - config.keep_last_epochs}.pt"
        if del_file.exists():
            del_file.unlink()
    
    return stats

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


# #  &#x1F4CC; 开始训练！
# ## 训练循环

# In[66]:


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


# # 提交预测结果

# In[69]:


# 平均几个检查点checkpoints可以产生与ensemble类似的效果
checkdir=config.savedir
get_ipython().system('python ./fairseq/scripts/average_checkpoints.py  --inputs {checkdir}  --num-epoch-checkpoints 5  --output {checkdir}/avg_last_5_checkpoint.pt')


# ## 确认用于生成提交的模型权重

# In[ ]:


# # checkpoint_last.pt : 最后个 epoch的保存点
# # checkpoint_best.pt : 验证集中 bleu 最高的保存点
# # avg_last_5_checkpoint.pt:　过去5个epoched的平均值
# try_load_checkpoint(model, name="avg_last_5_checkpoint.pt")
# validate(model, task, criterion, log_to_wandb=False)


# ## 生成预测结果

# In[70]:


def generate_prediction(model, task, split="test", outfile="./prediction.txt"):    
    task.load_dataset(split=split, epoch=1)
    itr = load_data_iterator(task, split, 1, config.max_tokens, config.num_workers).next_epoch_itr(shuffle=False)
    
    idxs = []
    hyps = []

    model.eval()
    progress = tqdm(itr, desc=f"prediction")
    with torch.no_grad():
        for i, sample in enumerate(progress):
            # 验证集损失
            sample = utils.move_to_cuda(sample, device=device)

            # 模型推理
            s, h, r = inference_step(sample, model)
            
            hyps.extend(h)
            idxs.extend(list(sample['id']))
            
    # 根据预处理前的顺序排序
    hyps = [x for _,x in sorted(zip(idxs,hyps))]
    
    with open(outfile, "w") as f:
        for h in hyps:
            f.write(h+"\n")


# In[72]:


generate_prediction(model, task)


# In[74]:


get_ipython().system('head -n 2 /kaggle/working/DATA/rawdata/ted2020/test.en')
get_ipython().system('head -n 2 prediction.txt')


# In[ ]:





# # 反向翻译`Back-translation`
# ## 训练反向翻译模型
# 1. 在**config中切换source_lang和target_lang** 
# 2. 更改**config**中的savedir（例如“./checkpoints/transformer back”）
# 3. 训练模型
# 
# ## 使用反向模型生成合成数据
# ### 下载 monolingual 数据

# In[ ]:


mono_dataset_name = 'mono'


# In[ ]:


mono_prefix = Path(data_dir).absolute() / mono_dataset_name
mono_prefix.mkdir(parents=True, exist_ok=True)

urls = (
    "https://github.com/yuhsinchan/ML2022-HW5Dataset/releases/download/v1.0.2/ted_zh_corpus.deduped.gz"
)
file_names = (
    'ted_zh_corpus.deduped.gz',
)

for u, f in zip(urls, file_names):
    path = mono_prefix/f
    if not path.exists():
        else:
            get_ipython().system('wget {u} -O {path}')
    else:
        print(f'{f} is exist, skip downloading')
    if path.suffix == ".tgz":
        get_ipython().system('tar -xvf {path} -C {prefix}')
    elif path.suffix == ".zip":
        get_ipython().system('unzip -o {path} -d {prefix}')
    elif path.suffix == ".gz":
        get_ipython().system('gzip -fkd {path}')


# ### <font color=darkred><b>***TODO***: 数据清洗(clean corpus)</font></b>
# 
# 1. 删除过长或过短的句子
# 2. 统一标点符号
# 
# 提示: 你可以使用上述定义的 `clean_s()` 来完成该操作 

# In[ ]:





# ###  <font color=darkred><b>***TODO***: 子串(Subword Units)</font></b>
# 
# 使用反向模型的spm模型将数据标记为子字单位
# 
# 提示: spm 模型本地位置 DATA/raw-data/\[dataset\]/spm\[vocab_num\].model

# In[ ]:





# ### 二进制化
# 
# 使用 fairseq 将数据转成二进制

# In[ ]:


binpath = Path('./DATA/data-bin', mono_dataset_name)
src_dict_file = './DATA/data-bin/ted2020/dict.en.txt'
tgt_dict_file = src_dict_file
monopref = str(mono_prefix/"mono.tok") # whatever filepath you get after applying subword tokenization
if binpath.exists():
    print(binpath, "exists, will not overwrite!")
else:
    get_ipython().system("python -m fairseq_cli.preprocess         --source-lang 'zh'         --target-lang 'en'         --trainpref {monopref}         --destdir {binpath}         --srcdict {src_dict_file}         --tgtdict {tgt_dict_file}         --workers 2")


# ### <font color=darkred><b>***TODO***: 使用反向模型生成合成数据</font></b> 
# 
# 将二进制`monolingual data`添加到原始数据目录，并使用“split_name”命名
# 比如： . ./DATA/data-bin/ted2020/\[split_name\].zh-en.\["en", "zh"\].\["bin", "idx"\]
# 
# 然后，你可以使用 `generate_prediction(model, task, split="split_name")` 去生成翻译预测

# In[ ]:


get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.zh.bin ./DATA/data-bin/ted2020/mono.zh-en.zh.bin')
get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.zh.idx ./DATA/data-bin/ted2020/mono.zh-en.zh.idx')
get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.en.bin ./DATA/data-bin/ted2020/mono.zh-en.en.bin')
get_ipython().system('cp ./DATA/data-bin/mono/train.zh-en.en.idx ./DATA/data-bin/ted2020/mono.zh-en.en.idx')


# In[ ]:





# # 参考
# 
# 1. <a name=ott2019fairseq></a>Ott, M., Edunov, S., Baevski, A., Fan, A., Gross, S., Ng, N., ... & Auli, M. (2019, June). fairseq: A Fast, Extensible Toolkit for Sequence Modeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics (Demonstrations) (pp. 48-53).
# 2. <a name=vaswani2017></a>Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017, December). Attention is all you need. In Proceedings of the 31st International Conference on Neural Information Processing Systems (pp. 6000-6010).
# 3. <a name=reimers-2020-multilingual-sentence-bert></a>Reimers, N., & Gurevych, I. (2020, November). Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP) (pp. 4512-4525).
# 4. <a name=tiedemann2012parallel></a>Tiedemann, J. (2012, May). Parallel Data, Tools and Interfaces in OPUS. In Lrec (Vol. 2012, pp. 2214-2218).
# 5. <a name=kudo-richardson-2018-sentencepiece></a>Kudo, T., & Richardson, J. (2018, November). SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations (pp. 66-71).
# 6. <a name=sennrich-etal-2016-improving></a>Sennrich, R., Haddow, B., & Birch, A. (2016, August). Improving Neural Machine Translation Models with Monolingual Data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 86-96).
# 7. <a name=edunov-etal-2018-understanding></a>Edunov, S., Ott, M., Auli, M., & Grangier, D. (2018). Understanding Back-Translation at Scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 489-500).
# 8. https://github.com/ajinkyakulkarni14/TED-Multilingual-Parallel-Corpus
# 9. https://ithelp.ithome.com.tw/articles/10233122
# 10. https://nlp.seas.harvard.edu/2018/04/03/attention.html
# 11. https://colab.research.google.com/github/ga642381/ML2021-Spring/blob/main/HW05/HW05.ipynb

# In[ ]:




