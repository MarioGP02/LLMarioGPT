import torch
import torch.nn as nn
from torch.nn import functional as F

# Hiperparámetros - Ajusta según tu GPU
batch_size = 32      # Cuántas secuencias procesamos a la vez
block_size = 192      # Longitud máxima de contexto (ventana de atención)
max_iters = 5000     # Cuánto tiempo entrenaremos
eval_interval = 250  # Cada cuánto ver el progreso
learning_rate = 5e-4 # Tasa de aprendizaje
device = 'cuda' if torch.cuda.is_available() else 'cpu' # ¡Crucial para local!

n_head = 6           # Número de cabezas de atención
head_size = 64       # Dimensión de cada cabeza
n_embd = n_head * head_size # Dimensión de los vectores de "pensamiento"
n_layer = 6          # Cuántos bloques Transformer apilamos
dropout = 0.2        # Para evitar que el modelo solo memorice

print(f"🚀 Entrenando en: {device.upper()}")

# Cargamos el dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# Mapeo de caracteres a enteros
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] 
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # 90% entrenamiento, 10% validación
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_ptr = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_ptr) - block_size, (batch_size,))
    x = torch.stack([data_ptr[i:i+block_size] for i in ix])
    y = torch.stack([data_ptr[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

class Head(nn.Module):
    """ Una sola cabeza de auto-atención """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) 
        q = self.query(x)
        # Calcula afinidades (atención)
        head_size = k.size(-1)
        wei = q @ k.transpose(-2,-1) * head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    """ Múltiples cabezas de atención calculadas en estricto PARALELO """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        # n_embd total es el número de cabezas multiplicado por el tamaño de cada una
        n_embd = num_heads * head_size 
        
        # MAGIA: Una sola capa lineal genera Q, K y V para TODAS las cabezas de un solo golpe
        # (Por eso multiplicamos por 3)
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size() # Batch Size, Time (block_size), Channels (n_embd)
        
        # 1. Calculamos Q, K y V al mismo tiempo
        qkv = self.c_attn(x)
        
        # 2. Los separamos en tres tensores distintos
        q, k, v = qkv.split(C, dim=2)
        
        # 3. Redimensionamos para "crear" las cabezas lógicamente y transponemos
        # Pasamos de forma (B, T, C) a (B, num_heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # 4. Atención ultra rápida y optimizada (Reemplaza a la máscara tril y el Softmax manual)
        # is_causal=True asegura que el modelo no haga "trampa" mirando al futuro
        y = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=dropout if self.training else 0.0, 
            is_causal=True
        )
        
        # 5. Volvemos a unir todas las cabezas en una sola línea de canales
        # .contiguous() es necesario en PyTorch después de un .transpose() antes de usar .view()
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 6. Proyección final
        return self.dropout(self.c_proj(y))

class FeedForward(nn.Module):
    """ Una capa lineal seguida de una no-linealidad (ReLU/GELU) """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Bloque Transformer: comunicación + computación """
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection
        x = x + self.ffwd(self.ln2(x))
        return x

class MarioLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding_table(pos)       # (1, T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

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
            temperature = 0.8
            top_k = 50

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
model = MarioLLM().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 1. Definimos la herramienta de evaluación UNA VEZ, fuera del bucle
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = {'train': [], 'val': []}
    for split in ['train', 'val']:
        for _ in range(10): # Promediamos sobre 10 batches para tener una buena estimación
            xb, yb = get_batch(split)
            _, loss = model(xb, yb)
            losses[split].append(loss.item())
    model.train()
    return {k: sum(v)/len(v) for k, v in losses.items()}

# 2. El bucle de entrenamiento se vuelve mucho más limpio
for iter in range(max_iters):
    
    # Cada 500 iteraciones (eval_interval), paramos a revisar
    if iter % eval_interval == 0:
        # ¡AQUÍ ESTÁ LA CLAVE! Llamamos a la función y guardamos el resultado
        losses = estimate_loss() 
        print(f"Iteración {iter} | Pérdida Train: {losses['train']:.4f} | Pérdida Val: {losses['val']:.4f}")

    # --- El entrenamiento normal continúa aquí ---
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generar algo de texto después del entrenamiento
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\n--- TEST DE GENERACIÓN ---")
model.eval()
with torch.no_grad():
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
