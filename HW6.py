import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16
block_size = 16
max_iters = 2000
learning_rate = 2e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
embedding_dim = 64
hidden_dim = 128
num_layers = 2

text = """
循環神經網路與長短期記憶網路是傳統語言模型的基石。
它們不需要自注意力機制，而是透過時間序列一步步傳遞隱藏狀態。
雖然現在Transformer很流行，但非注意力模型在輕量化設備上依然很有優勢。
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_int = { ch:i for i,ch in enumerate(chars) }
int_to_char = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [char_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_char[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class RNNLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, idx, targets=None):
        embedded = self.embedding(idx)
        out, _ = self.rnn(embedded)
        logits = self.lm_head(out)

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
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = RNNLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

start_context = torch.tensor([encode("循")], dtype=torch.long, device=device)
generated_text = decode(model.generate(start_context, max_new_tokens=40)[0].tolist())
print(generated_text)
