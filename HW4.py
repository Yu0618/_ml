import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. 超參數設定 (為了能在筆電本地快速執行，設定為微型版本)
batch_size = 16       # 同時處理的句子數量
block_size = 32       # 文本最大長度 (Context Length)
max_iters = 1000      # 訓練迭代次數
eval_interval = 200   # 每隔幾次評估一次損失
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64           # 嵌入維度 (Embedding dimension)
n_head = 4            # 多頭注意力的頭數 (Multi-head attention)
n_layer = 4           # Transformer Block 的層數

print(f"目前使用裝置: {device}")

# 2. 準備訓練文本 (這裡用一段簡單的文本讓它學習語感)
text = """
人工智慧是未來的趨勢。大語言模型可以幫助我們寫程式、回答問題。
開發一個自己的GPT並不難，只要掌握Transformer的核心原理。
今天的天氣很好，我們一起來學習機器學習與深度學習吧。
"""

# 建立字符映射表 (Character-level Tokenizer)
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s] # 字串轉數字
decode = lambda l: ''.join([int_to_char[i] for i in l]) # 數字轉字串

# 將文本轉換為 Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# 3. 隨機獲取批次數據的函數
def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# 4. GPT 核心組件：單頭注意力 (Head)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # 計算注意力權重 (Scales Attention Scores)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        # 因果遮罩 (Causal Mask)：確保模型看不到未來的字
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

# 5. 多頭注意力 (Multi-Head Attention)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

# 6. 前饋神經網路 (Feed Forward)
class FeedFoward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

# 7. Transformer 區塊 (Block)
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))   # 殘差連接與層歸一化
        x = x + self.ffwd(self.ln2(x))
        return x

# 8. 完整的微型 GPT 模型
class MiniGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # 裁剪上下文到最大長度
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # 只關注最後一個字的預測
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # 依機率抽樣下一個字
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# 9. 初始化模型與優化器
model = MiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 10. 訓練迴圈
print("開始訓練微型 GPT...")
for iter in range(max_iters):
    if iter % eval_interval == 0:
        xb, yb = get_batch()
        logits, loss = model(xb, yb)
        print(f"步數 {iter:4d} | 訓練損失 (Loss): {loss.item():.4f}")

    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("-" * 40)
print("訓練完成！嘗試讓本地 GPT 自動生成文本：")

# 11. 測試模型生成能力
context = torch.zeros((1, 1), dtype=torch.long, device=device) # 以空字元（或預設第一個字）開始接龍
generated_text = decode(model.generate(context, max_new_tokens=50)[0].tolist())
print(generated_text)
