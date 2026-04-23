import torch
import torch.nn as nn
import torch.nn.functional as F

with open('input.txt', 'r') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
n_emb = 32
print(f'vocab size: {vocab_size}')

stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for i,s in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
batch_size = 32

torch.manual_seed(1337)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint((len(data) - block_size), (batch_size,))
    # print(f"ix: {ix.shape}")
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    return x, y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_emb, head_size, bias=False)
        self.query = nn.Linear(n_emb, head_size, bias=False)
        self.value = nn.Linear(n_emb, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5                         # basically ye btata ki kis token ko kis token pe kitna dhyan dena chahiye
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # yha par tril ka use isliye kiya hai taki future tokens pe dhyan na de
        wei = F.softmax(wei, dim=-1)                                    #yha par softmax ka use isliye kiya hai taki attention scores ko probabilities me convert kar sake

        v = self.value(x) # (B, T, hs)
        out = wei @ v # (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_emb, n_emb)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_emb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4*n_emb),
            nn.ReLU(),
            nn.Linear(4*n_emb, n_emb)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_emb, num_heads=4):
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads=num_heads, head_size=n_emb//num_heads)
        self.ffwd = FeedForward(n_emb)

    def forward(self, x):
        x = x + self.sa_heads(x)
        x = x + self.ffwd(x)
        return x

torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_emb)            # lookup table for token embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_emb)         # lookup table for position embeddings
        self.blocks = nn.Sequential(
            Block(n_emb, num_heads=4),
            Block(n_emb, num_heads=4),
            Block(n_emb, num_heads=4),
        )                                          # feed forward network
        self.lm_head = nn.Linear(n_emb, vocab_size)   # ye final layer ha jo hume logits dega har token ke liye ki agla token kya ho sakta hai

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C)           # emb table ma lookup karke hume token embeddings mil jayenge
        pos_emb = self.position_embedding_table(torch.arange(T))        # position embeddings bhi lookup table se mil jayenge
        x = tok_emb + pos_emb                                           # yha par token embeddings aur position embeddings ko add kar diya hai taki hume dono ki information mil jaye
        x = self.blocks(x)                             
        logits = self.lm_head(x) # (B, T, vocab_size)     # yha par final layer se hume logits mil jayenge har token ke liye ki agla token kya ho sakta hai

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
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel()
# out, loss = m(xb, yb)
# print(out.shape)

# ix = torch.zeros((1, 1), dtype=torch.long)
# print(decode(m.generate(ix, max_new_tokens=1)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(10000):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    print(f'est_loss {loss.item()}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
