import math
import torch
from torch import nn
from torch.nn import functional as F
from .embed import Embedding

# SwiGLU 激活函数模块 一种用于前馈层的替代激活函数，比传统 ReLU 更灵活。
class SwiGLU(nn.Module):
    # 将输入 x 沿最后一个维度均等地拆分为两部分：一部分为主信息，另一部分为门控信号
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # 使用 SiLU 激活函数处理门控信号，然后与主信息逐元素相乘
        return F.silu(gate) * x


class CausalSelfAttention(nn.Module):
    # 因果自注意力模块（CausalSelfAttention）
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    实现了多头自注意力机制，并对未来的 token 进行遮掩（因果遮掩），
    确保生成时每个 token 只能依赖于其之前的 token。
    实现带掩码的多头自注意力机制
    """
    def __init__(self, config):
        super().__init__()
        # 验证嵌入维度是否能被头数整除
        assert config.N_EMBD % config.N_HEADS == 0
        # key, query, value projections for all heads
        # 定义线性层，将输入映射为 key, query, value 向量，维度均为 N_EMBD
        self.key = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.query = nn.Linear(config.N_EMBD, config.N_EMBD)
        self.value = nn.Linear(config.N_EMBD, config.N_EMBD)
        # regularization 正则化
        # dropout，用于正则化注意力矩阵和残差连接
        self.attn_drop = nn.Dropout(config.ATTN_PDROP)
        self.resid_drop = nn.Dropout(config.RESID_PDROP)
        # output projection
        # 输出投影层，将多头注意力的结果恢复到 N_EMBD 维度
        self.proj = nn.Linear(config.N_EMBD, config.N_EMBD)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # 构造因果遮掩矩阵，确保每个 token 只能关注自己之前的信息。
        # 使用三角下半部分矩阵，尺寸为 (1, 1, MAX_LEN+NUM_PROPS, MAX_LEN+NUM_PROPS)
        self.register_buffer("mask", torch.tril(torch.ones(config.MAX_LEN+config.NUM_PROPS, config.MAX_LEN+config.NUM_PROPS))
                                     .view(1, 1, config.MAX_LEN+config.NUM_PROPS, config.MAX_LEN+config.NUM_PROPS))
        self.n_head = config.N_HEADS

    def forward(self, x):
        B, T, C = x.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # 通过 key, query, value 线性层计算，并重塑为 (B, n_head, T, head_size)
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # 计算注意力得分: 用 query 和 key 做点积，再按 head_size 开根号缩放
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        # Softmax 归一化，得到注意力权重
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # 计算加权求和：注意力权重乘以 value 得到输出
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # 将多头输出重新组合回原始维度 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection 最后经过输出投影和残差 dropout 返回结果
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block
    Transformer 模块的基本 Block，由 LayerNorm、因果自注意力和前馈网络组成"""
    def __init__(self, config):
        super().__init__()
        # 定义两个 LayerNorm，分别用于注意力层和 MLP 层前的标准化
        self.ln1 = nn.LayerNorm(config.N_EMBD)
        self.ln2 = nn.LayerNorm(config.N_EMBD)
        # 定义因果自注意力模块
        self.attn = CausalSelfAttention(config)
        # 定义前馈网络（MLP）：两层全连接+ GELU 激活+ Dropout
        self.mlp = nn.Sequential(
            nn.Linear(config.N_EMBD, 4 * config.N_EMBD),
            nn.GELU(),
            nn.Linear(4 * config.N_EMBD, config.N_EMBD),
            nn.Dropout(config.RESID_PDROP))

    def forward(self, x):
        # 先对输入 x 进行 LayerNorm 标准化，然后计算注意力输出
        y = self.ln1(x)
        y = self.attn(y)
        # 第一处残差连接：将注意力模块的输出与原始输入相加
        x = x + y
        # 对更新后的 x 再次进行 LayerNorm 后送入 MLP，之后再次加上残差连接
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size
    完整的 GPT 模型，负责从输入的 token 序列生成输出（例如 SMILES 序列） """
    def __init__(self, config):
        super().__init__()
        self.device = config.DEVICE
        self.config = config.MODEL
        # 构造多个 Transformer Block，堆叠成一个序列
        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.N_LAYERS)])
        # 嵌入层，将 token（以及可能的属性）转换为向量表示
        self.embed = Embedding(config=self.config, device=self.device)
        # 最后一个 LayerNorm，用于稳定最终输出
        self.ln_f = nn.LayerNorm(self.config.N_EMBD)
        # 输出层，将内部表示映射回词汇表尺寸，输出 logits 用于概率预测
        self.head = nn.Linear(self.config.N_EMBD, self.config.VOCAB_SIZE, bias=False)
        self.block_size = self.config.MAX_LEN
        # 初始化权重
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        # 对 Linear 层与 Embedding 进行正态分布初始化
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        # 对 LayerNorm 的权重和偏置进行初始化
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, targets=None, prop=None):
        # 将输入 token (idx) 与附加属性 (prop) 通过嵌入层转换为向量表示
        x = self.embed(token=idx, prop=prop)
        # 逐层经过 Transformer Block
        x = self.blocks(x)
        # 最后经过 LayerNorm
        x = self.ln_f(x)
        # 生成预测 logits（每个 token 对应的词汇表概率分布）
        logits = self.head(x)
        # 若存在额外的属性数量，则进行裁剪（具体实现与模型设置有关）
        if self.config.NUM_PROPS:
            logits = logits[:, self.config.NUM_PROPS:, :]
        loss = None
        # 如果提供了 target 标签，则计算交叉熵损失
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

#  模型采样函数：逐步生成 token
@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample_=False, top_k=None, prop=None):
    """
    给定一个条件序列 x（形状 (batch, t)），预测后续 steps 个 token，
    并将预测出的 token 逐步附加到序列上，实现自动续写。模型
    在每一步使用当前序列（裁剪至 block_size）进行预测。
    """
    block_size = model.get_block_size()   
    model.eval()
    for k in range(steps):
        # 如果当前序列超过 block_size，则截取最近的部分    crop context if needed
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _ = model(x_cond, prop = prop)   #   通过模型预测 logits
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # 如果设定了 top_k，则对 logits 进行截断，只保留 top_k 值
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        # 经过 softmax 将 logits 转换为概率分布
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        # 根据设置选择采样方式：随机采样或贪心选择
        if sample_:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        # 将新预测的 token 附加到当前序列上
        x = torch.cat((x, ix), dim=1)
    return x




# take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
#     the sequence, feeding the predictions back into the model each time. Clearly the sampling
#     has quadratic complexity unlike an RNN that is only linear, and has a finite context window
#     of block_size, unlike an RNN that has an infinite context window.
# optionally crop probabilities to only the top k options