import os
from typing import Dict, Optional

import wget
import torch
import numpy as np
import random

from torch import Tensor

torch.backends.cudnn.benchmark = True
# 随机种子
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

# 数据处理
# 全角转半角：全角占两个字符宽度，半角占一个
def strQ2B(ustring):
    """
    全角转半角
    """
    ss = []
    for s in ustring:
        for uchar in s:
            unicode_point = ord(uchar)
            if unicode_point == 12288:
                unicode_point = 32
            elif (unicode_point >= 65281 and unicode_point <= 65374):   # 全角字符（除空格）根据关系转化，>65281并且<65374
                unicode_point -= 65248# 转换
            rstring = chr(unicode_point)
        ss.append(rstring)
    return ''.join(ss)

import re
def clean_s(s, lang):# 清理数据函数
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

def len_s(s, lang):
    if lang == 'zh':
        return len(s)
    return len(s.split())

from pathlib import Path

prefix = Path(r'D:\ML\data\ted2020')

from IPython import get_ipython
from tqdm import tqdm
# o=[prefix ,"prefix:"]; print(o[1], o[0])
def show_file():
    num_line = sum(1 for line in open(f'{prefix}/train_dev.raw.en','r',encoding='utf-8'))
    o=[num_line ,"num_line:"]; print(o[1], o[0])

    # with open(f'{prefix}/train_dev.raw.en','r',encoding='utf-8') as f:
        # for line in tqdm(f):
            # o=[line ,"line:"]; print(o[1], o[0])

# show_file()
# 文件数据清理函数：当前文件名为：train_dev.raw.en，清理后的文件名为train_dev.raw.clean.en
origin_train_en_file = Path(prefix)/'train_dev.raw.en'
origin_train_zh_file = Path(prefix)/'train_dev.raw.zh'

origin_test_en_file = Path(prefix)/'train_dev.raw.en'
origin_test_zh_file = Path(prefix)/'train_dev.raw.zh'


cleaned_train_zh_file = Path(prefix)/'train_dev.raw.clean.zh'
cleaned_train_en_file = Path(prefix)/'train_dev.raw.clean.en'


cleaned_test_zh_file = Path(prefix)/'test.raw.clean.zh'
cleaned_test_en_file = Path(prefix)/'test.raw.clean.en'
def clean_corpus(data_type, prefix, ratio=9, max_len=1000, min_len=1):
    if data_type == 'train':
        cleaned_zh_file = cleaned_train_zh_file
        cleaned_en_file = cleaned_train_en_file
    elif data_type == 'test':
        cleaned_zh_file = cleaned_test_zh_file
        cleaned_en_file = cleaned_test_en_file
    if cleaned_zh_file.exists() and cleaned_en_file.exists():
        print(f"{cleaned_zh_file} is exists, skip clean")
        return
    with open(f'{origin_train_en_file}','r', encoding='utf-8')as origin_en_file:
        with open(f'{origin_train_zh_file}','r',encoding='utf-8')as origin_zh_file:
            with open(f'{cleaned_en_file}','w',encoding='utf-8')as cleaned_en_file:
                with open(f'{cleaned_zh_file}','w',encoding='utf-8')as cleaned_zh_file:
                    for origin_en_line in origin_en_file:
                        orgin_zh_line = origin_zh_file.readline()
                        origin_en_line = origin_en_line.strip()
                        orgin_zh_line = orgin_zh_line.strip()
                        clean_s(origin_en_line, 'en')
                        clean_s(orgin_zh_line, 'zh')
                        zh_len = len_s(orgin_zh_line,'zh')
                        en_len = len_s(origin_en_line,'en')
                        if max_len > 0:
                            if zh_len > max_len or en_len > max_len:
                                continue
                        if min_len > 0:
                            if zh_len < min_len or en_len < min_len:
                                continue
                        print(orgin_zh_line, file=cleaned_zh_file)
                        print(origin_en_line, file=cleaned_en_file)

# 数据清理
def clean_data():
    clean_corpus(data_type='train', prefix=prefix)
    clean_corpus(data_type='test', prefix=prefix)

# 拆分训练集和验证集
splited_train_file_en = Path(prefix)/'splited.train.clean.en'
splited_train_file_zh = Path(prefix)/'splited.train.clean.zh'

splited_valid_file_en= Path(prefix)/ 'splited.valid.clean.en'
splited_valid_file_zh = Path(prefix)/'splited.valid.clean.zh'

def splite_data():
    valid_ratio = 0.01
    train_ratio = 1-valid_ratio
    global prefix

    if splited_valid_file_zh.exists() and splited_train_file_en.exists() and splited_train_file_zh.exists() and splited_valid_file_en.exists():
        print("splited data set is exists, skipping splite")

    for cleaned_file in [cleaned_train_en_file, cleaned_train_zh_file]:
        num_line = sum(1 for line in open(f'{cleaned_file}','r',encoding='utf-8'))
        labels = list(range(num_line))
        random.shuffle(labels)
        if cleaned_file == cleaned_train_en_file:
            train_splited_file = open(f'{splited_train_file_en}','w',encoding='utf-8')
            valid_splited_file = open(f'{splited_valid_file_en}','w',encoding='utf-8')
        else:
            train_splited_file = open(f'{splited_train_file_zh}','w',encoding='utf-8')
            valid_splited_file = open(f'{splited_valid_file_zh}','w',encoding='utf-8')
        count = 0
        for line in open(f'{cleaned_file}','r',encoding='utf-8'):
            if labels[count] / num_line < train_ratio:# 随机选行加入训练集，只要比例够就行
                train_splited_file.write(line)
            else:
                valid_splited_file.write(line)
            count+=1
        train_splited_file.close()
        valid_splited_file.close()

import sentencepiece as spm
vocab_size = 8000
def train_vacab():
    if (prefix/ f'spm{vocab_size}.model').exists():
        print("vocab model exists, skipping train")
        return
    # o=[f'{splited_train_file_zh}' ,"f'{splited_train_file_zh}':"]; print(o[1], o[0])
    spm.SentencePieceTrainer.train(# 句子分词模型
        input=','.join([f'{splited_train_file_zh}',
                        f'{splited_train_file_en}',
                        f'{splited_valid_file_en}',
                        f'{splited_valid_file_zh}']),
        model_prefix=prefix/f'spm{vocab_size}',# 训练的模型存放地
        vocab_size=vocab_size,
        character_coverage=1,
        model_type='unigram', # 用'bpe'也行
        input_sentence_size=1e6,
        shuffle_input_sentence=True,
        normalization_rule_name='nmt_nfkc_cf',
    )

# splite_data()
# train_vacab()
# 使用分词器分词
spm_model = spm.SentencePieceProcessor(str(f'{prefix}/spm{vocab_size}.model'))
train_en = prefix/'train.en'
valid_en = prefix/'valid.en'
train_zh = prefix/'train.zh'
valid_zh = prefix/'valid.zh'
test_en = prefix/'test.en'
test_zh = prefix/'test.zh'

def tokenlize_file(source_file, target_file):
    with open(f'{source_file}', 'r', encoding='utf-8') as sf:
        with open(f'{target_file}', 'w', encoding='utf-8') as tf:
            for line in sf:# 对行做tokenlize
                # 先拆分成列表
                line = line.strip()
                token = spm_model.encode(line, out_type=str)
                print(' '.join(token), file=tf)

def tokenlize_files():
    tokenlize_file(splited_train_file_en, train_en)
    tokenlize_file(splited_train_file_zh, train_zh)
    tokenlize_file(splited_valid_file_en, valid_en)
    tokenlize_file(splited_valid_file_zh, valid_zh)
    tokenlize_file(cleaned_test_en_file, test_en)
    tokenlize_file(cleaned_test_zh_file, test_zh)


from argparse import Namespace
# 设置超参数
config = Namespace(# 超参数
    datadir = r"D:\ML\data\ted2020",
    savedir = r"D:\ML\models",
    source_lang = "en",
    target_lang = "zh",

    # 设置cpu核数：fetching & processing data.
    num_workers=2,
    # batch size 中tokens大小设置. 梯度累积增加有效批量 gradient accumulation increases the effective batchsize.
    max_tokens=8192,
    accum_steps=2,

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
# 日志系统
import logging
import sys
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",# 形式
    datefmt="%Y-%m-%d %H:%M:%S",# 时间格式
    level="INFO", # "DEBUG" "WARNING" "ERROR"# 日志级别
    stream=sys.stdout,# 输出到何处
)
proj = "seq2seq"# 项目名称
logger = logging.getLogger(proj)# 创建日志记录器
if config.use_wandb:
    import wandb
    wandb.init(project=proj, name=Path(config.savedir).stem, config=config)

# 使用fairseq加载数据集
from fairseq import utils
cuda_env = utils.CudaEnvironment()# 设置cuda环境
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')# 设置计算设备
device = 'cpu'
logger.info("for epoch")
from fairseq.tasks.translation import TranslationConfig, TranslationTask# 导入翻译配置和翻译认为模块，这个模块就是一个现成的翻译模型

task_cfg = TranslationConfig(# 翻译配置
    data=config.datadir,# 数据路径
    source_lang=config.source_lang,# 原语言
    target_lang=config.target_lang,# 目标语言
    train_subset="train",# 训练集名称
    required_seq_len_multiple=8,#
    dataset_impl="mmap",# 加载数据集的方式，使用mmap映射到进程内存
    upsample_primary=1,# 是否对主数据集采样
)
# 将数据二进制化
import subprocess

# command = ['python -m fairseq_cli.preprocess', '--source-lang', 'en', '--target-lang',  'zh', '--trainpref ', f'{prefix}/train', '--validpref',f'{prefix}/valid'         ,'--testpref',f'{prefix}/test','--destdir',f'{prefix}', '--joined-dictionary',        '--workers 2']

# o=[command ,"command:"]; print(o[1], o[0])

# subprocess.run(command)
task = TranslationTask.setup_task(task_cfg)

task.load_dataset(split='train', epoch=1, combine=True)
task.load_dataset(split='valid', epoch=1)

seed=config.seed
def load_data_iterator(task, split, epoch=1, max_tokens=4000, num_workers=1, cached=True):
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

demo_epoch_obj = load_data_iterator(task, 'valid', epoch=1, max_tokens=20, num_workers=1,cached=False)
demo_iter = demo_epoch_obj.next_epoch_itr(shuffle=True)
# o=[next(demo_iter) ,"next(demo_iter):"]; print(o[1], o[0])

import torch.nn as nn
# 构建rnn编码器
from fairseq.models import FairseqEncoder, FairseqIncrementalDecoder, FairseqEncoderDecoderModel
class RNNEncoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(RNNEncoder, self).__init__(dictionary)
        self.embed_tokens = embed_tokens
        self.embed_dim = args.encoder_embed_dim
        self.hidden_dim = args.encoder_ffn_embed_dim
        self.num_layers = args.encoder_layers

        self.dropout_in_moudle = nn.Dropout(args.dropout)

        self.rnn = nn.GRU(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=args.dropout,
            batch_first=False,
            bidirectional=True
        )

        self.padding_idx = dictionary.pad()

    def combine_bidir(self, outs, bsz:int):# 合并双向隐藏层
        outs.view(self.num_layers, 2, bsz, -1).transpose(1,2).contiguous()
        return outs.view(self.num_layers, bsz, -1)

    def forward(self, src_tokens, **unused):
        bsz, seq_len = src_tokens.size()

        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_moudle(x)

        # 转换成序列长度×批量×序列维度
        x = x.transpose(0,1)

        h0 = x.new_zeros(2*self.num_layers, bsz, self.hidden_dim)
        x,final_hidden = self.rnn(x, h0)

        outputs = self.dropout_in_moudle(x)
        final_hidden = self.combine_bidir(final_hidden, bsz)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()
        return tuple(
            (
                outputs,
                final_hidden,
                encoder_padding_mask
            )
        )

    # 用于束搜索的函数
    def reorder_encoder_out(self, encoder_out, new_order):
        return tuple(
            (
                encoder_out[0].index_select(1, new_order),
                encoder_out[1].index_select(1, new_order),
                encoder_out[2].index_select(1, new_order),
            )
        )
    def forward1(self, src_tokens, **unused):
        bsz, seq_len = src_tokens.size()

        x = self.embed_tokens(src_tokens)
        x = self.dropout_in_moudle(x)
        x=x.transpose(0,1)

        h0 = x.new_zeros(2*self.num_layers, bsz, self.hidden_dim)
        x, final_hidden  = self.rnn(x, h0)

        outputs = self.dropout_in_moudle(x)
        final_hidden = self.combine_bidir(final_hidden, bsz)

        encode_pad_mask = src_tokens.eq(self.padding_idx).t()
        return tuple(
            (
                outputs,
                final_hidden,
                encode_pad_mask
            )
        )

from torch.nn import functional as F

# 构建注意力层
class AttentionLayer(nn.Module):
    def __init__(self, decoder_input_embed_dim, encoder_output_embed_dim, output_embed_dim, bias=False):
        super(AttentionLayer, self).__init__()
        self.Q = nn.Linear(decoder_input_embed_dim, encoder_output_embed_dim, bias=bias)
        self.output_projection = nn.Linear(decoder_input_embed_dim+encoder_output_embed_dim, output_embed_dim, bias=bias)

    def K(self, input):
        return input

    def V(self, input):
        return input

    def forward(self, decoder_inputs, encoder_outputs, encoder_padding_mask):
        decoder_inputs = decoder_inputs.transpose(1,0)
        encoder_outputs = encoder_outputs.transpose(1,0)
        encoder_padding_mask = encoder_padding_mask.transpose(1,0)# SB

        x = self.Q(decoder_inputs)# BTD
        k = self.K(encoder_outputs.transpose(1,2))# BSD-》BDS
        attention_weights = torch.bmm(x, k)# BTD×BDS=BTS
        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.unsqueeze(1)# B1S
            attention_weights = (attention_weights.float().masked_fill_(encoder_padding_mask, float("-inf")).type_as(attention_weights))

        attention_weights = F.softmax(attention_weights, dim=-1)# 最后一个维度上计算softmax，也就是每一行计算一个
        v = self.V(encoder_outputs)
        x = torch.bmm(attention_weights, v)
        x = torch.cat((x, decoder_inputs), dim=-1)# 融合解码注意力和输出
        x = torch.tanh(self.output_projection(x))
        return x.transpose(1,0), attention_weights

# 构建rnn解码器
class RNNDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, dictionary, embed_tokens):
        super(RNNDecoder, self).__init__(dictionary)
        self.embed_token = embed_tokens

        assert args.decoder_layers == args.decoder_layers
        assert args.decoder_ffn_embed_dim == 2*args.encoder_ffn_embed_dim

        self.embed_dim = args.decoder_embed_dim
        self.hidden_dim = args.decoder_ffn_embed_dim
        self.num_layers = args.decoder_layers

        self.dropout_in_module = nn.Dropout(args.dropout)
        self.rnn = nn.GRU(
            self.embed_dim,
            self.hidden_dim,
            self.num_layers,
            dropout=args.dropout,
            batch_first=False,
            bidirectional=False
        )

        self.attention = AttentionLayer(
            self.embed_dim, self.hidden_dim, self.embed_dim,bias=False
        )

        self.dropout_out_module = nn.Dropout(args.dropout)

        if self.hidden_dim != self.embed_dim:
            self.project_to_out_dim = nn.Linear(self.hidden_dim, self.embed_dim)
        else:
            self.project_to_out_dim = None

        # 输入嵌入和输出层使用相同参数
        if args.share_decoder_input_output_embed:
            self.output_project = nn.Linear(
                self.embed_token.weight.shape[0],# 权重的第一维
                self.embed_token.weight.shape[1],
                bias=False
            )
            self.output_project.weight = self.embed_token.weight
        else:
            self.output_project = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(self.output_project.weight, mean=0,std=self.output_embed_dim ** -0.5)# 有助于保持梯度稳定

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None, **unused):
        encoder_outputs, encoder_hiddens, encoder_padding_mask = encoder_out

        if incremental_state is not None and len(incremental_state)>0:
            prev_output_tokens = prev_output_tokens[:,-1:]# 最后一列
            cache_state = self.get_incremental_state(incremental_state, 'cached_state')
            prev_hiddens = cache_state["prev_hiddens"]
        else:
            prev_hiddens = encoder_hiddens

        bsz, seq_len = prev_output_tokens.size()

        x = self.embed_token(prev_output_tokens)
        x = self.dropout_in_module(x)
        x = x.transpose(0,1)

        if self.attention is not None:
            x, attn = self.attention(x, encoder_outputs, encoder_padding_mask)

        x, final_hiddens = self.rnn(x, prev_hiddens)
        x = self.dropout_out_module(x)
        if self.project_to_out_dim != None:
            x = self.project_to_out_dim(x)

        x = self.output_project(x)
        x = x.transpose(1,0)

        cache_state = {
            "prev_hiddens":final_hiddens,
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return x, None

    def reorder_incremental_state(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        cache_state = self.get_incremental_state(incremental_state, "cached_state")
        prec_hiddens = cache_state["prev_hiddens"]
        prec_hiddens = [p.index_select(0, new_order) for p in prec_hiddens]
        cache_state = {
            "prev_hiddens":torch.stack(prec_hiddens),
        }
        self.set_incremental_state(incremental_state, "cached_state", cache_state)
        return

# 构建序列到序列的模型
class seq2seq(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super(seq2seq, self).__init__(encoder, decoder)
        self.args = args

    def forward(self, src_tokens, src_lengths, prev_output_tokens, return_all_hiddens:bool=True):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens)
        out, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out, src_tokens=src_tokens, return_all_hiddens=return_all_hiddens)
        return out, extra

# 构建整个模型
def build_model(args, task):
    src_dict, tgt_dict = task.source_dictionary, task.target_dictionary
    encoder_embed_tokens = nn.Embedding(len(src_dict), args.encoder_embed_dim, src_dict.pad())
    decoder_embed_tokens = nn.Embedding(len(tgt_dict), args.decoder_embed_dim, tgt_dict.pad())
    encoder = RNNEncoder(args, src_dict, encoder_embed_tokens)
    decoder = RNNDecoder(args, tgt_dict, decoder_embed_tokens)

    model = seq2seq(args, encoder, decoder)

    def init_params(module):
        from fairseq.modules import MultiheadAttention
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02,)
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
    model.apply(init_params)
    return model

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
# 构建损失评估类
all_seed(73)
if config.use_wandb:
    wandb.config.update(vars(arch_args))


model = build_model(arch_args, task)
logger.info(model)


class LabelSmoothedCrossEntropyCriterion(nn.Module):# 用于替换标准交叉熵损失函数，避免对任何单一标签过于自信
    def __init__(self, smoothing, ignore_idx=None, reduce=True):
        super(LabelSmoothedCrossEntropyCriterion, self).__init__()
        self.smoothing = smoothing
        self.ignore_idx = ignore_idx
        self.reduce = reduce
        
    def forward(self, lprobs, target):
        if target.dim() == lprobs.dim()-1:
            target = target.unsqueeze(-1)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

        if self.ignore_idx is not None:
            pad_mask = target.eq(self.ignore_idx)# 计算等于填充值的元素
            nll_loss.masked_fill_(pad_mask, 0.0)
            smooth_loss.masked_fill_(pad_mask, 0.0)
        else:
            nll_loss = nll_loss.squeeze(-1)
            smooth_loss = smooth_loss.squeeze(-1)
        eps_i = self.smoothing / lprobs.size(-1)
        loss = (1-self.smoothing)*nll_loss + eps_i*smooth_loss
        return loss

criterion = LabelSmoothedCrossEntropyCriterion(
    smoothing=0.1,
    ignore_idx=task.target_dictionary.pad(),
)

def get_tate(d_model, step_num, warmup_step):
    lr = np.power(d_model, -0.5)*min(np.power(step_num, -0.5), step_num*np.power(warmup_step,-1.5))
    return lr

class NoamOpt:
    def __init__(self, model_size, factor, warmup,optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def multiply_grads(self,c):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def step(self):
        self._step+=1
        rate = self.rate()
        for p in self.param_groups:
            p['lr']=rate
        self._rate=rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return 0 if not  step else self.factor*get_tate(self.model_size,step, self.warmup)


import matplotlib.pyplot as plt
optimizer = NoamOpt(
    model_size=arch_args.encoder_embed_dim,
    factor=config.lr_factor,
    warmup=config.lr_warmup,
    optimizer=torch.optim.AdamW(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0001))

def show_lr():
    plt.title("lr Scheduling:\n $lrate = d_{\text{model}}^{-0.5}\cdot\min({step\_num}^{-0.5},{step\_num}\cdot{warmup\_steps}^{-1.5})$")
    plt.plot(np.arange(1, 100000), [optimizer.rate(i) for i in range(1, 100000)])
    plt.legend([f"{optimizer.model_size}:{optimizer.warmup}"])
    plt.show()

# show_lr()

from fairseq import utils
from fairseq.data import iterators
from torch.cuda.amp import GradScaler, autocast

def train_one_epoch(epoch_itr, model, task, criterion, optimizer, accum_steps=1):
    itr = epoch_itr.next_epoch_itr(shuffle=True)
    itr = iterators.GroupedIterator(itr, accum_steps)

    stats = {"loss": []}
    scaler = GradScaler()

    model.train()
    progress = tqdm(itr, desc=f"train epoch {epoch_itr.epoch}", leave=True)

    for samples in progress:
        model.zero_grad()
        accum_loss = 0
        sample_size = 0
        for i, sample in enumerate(samples):
            if i == 1:
                torch.cuda.empty_cache()
            sample = utils.move_to_cuda(sample, device=device)
            target = sample["target"]
            sample_size_i = sample["ntokens"]
            sample_size += sample_size_i

            with autocast():
                net_output = model.forward(**sample["net_input"])
                # o=[(sample["net_input"]) ,"sample[net_input]:"]; print(o[1], o[0])
                #
                # o=[net_output[0] ,"net_output[0]:"]; print(o[1], o[0])
                # o=[net_output[0].shape ,"net_output[0].shape:"]; print(o[1], o[0])

                lprobs = F.log_softmax(net_output[0], -1)
                # o=[lprobs.shape ,"lprobs.shape:"]; print(o[1], o[0])
                
                
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))
                # o=[loss ,"loss:"]; print(o[1], o[0])
                # o=[loss.shape ,"loss.shape:"]; print(o[1], o[0])

                accum_loss+=loss.mean().item()
                scaler.scale(loss.mean()).backward()

        scaler.unscale_(optimizer)
        optimizer.multiply_grads(1/(sample_size or 1.0))
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)

        scaler.step(optimizer)
        scaler.update()

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


sequence_generator = task.build_generator([model], config)# 使用task构建一个序列生成器

def decode(toks, dictionary):
    s = dictionary.string(
        toks.int().cpu(),
        config.post_process,
    )
    return s if s else "<unk>"

def inference_step(sample, model):
    gen_out = sequence_generator.generate([model], sample)
    srcs = []
    hyps = []
    refs = []

    for i in range(len(gen_out)):
        srcs.append(decode(
            utils.strip_pad(sample["net_input"]["src_tokens"][i], task.source_dictionary.pad()),
            task.source_dictionary,
        ))

        hyps.append(decode(
            gen_out[i][0]["tokens"],
            task.target_dictionary,
        ))

        refs.append(decode(
            utils.strip_pad(sample["target"][i],task.target_dictionary.pad()),
            task.target_dictionary,
        ))
    return  srcs, hyps, refs


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

if __name__ == "__main__":
    while epoch_itr.next_epoch_idx <= config.max_epoch:
        # train for one epoch
        train_one_epoch(epoch_itr, model, task, criterion, optimizer, config.accum_steps)
        stats = validate_and_save(model, task, criterion, optimizer, epoch=epoch_itr.epoch)
        logger.info("end of epoch {}".format(epoch_itr.epoch))
        epoch_itr = load_data_iterator(task, "train", epoch_itr.next_epoch_idx, config.max_tokens, config.num_workers)







