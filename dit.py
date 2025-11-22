from torch import nn
import torch as t

@t.no_grad()
def sample(model, z, y, n_steps=10, cfg=0):
  ts = t.linspace(1, 0, n_steps+1, device=z.device, dtype=z.dtype)
  ts = 3*ts / (2*ts+1) # sd3 scheduler
  for idx in range(n_steps):
    v_pred = model(z, y, ts[idx]*t.ones(z.shape[0], dtype=z.dtype, device=z.device))
    if cfg > 0:
      v_uncond = model(z, y*0, ts[idx]*t.ones(z.shape[0], dtype=z.dtype, device=z.device))
      v_pred = v_uncond + cfg*(v_pred - v_uncond)
    z = z + (ts[idx]-ts[idx+1])*v_pred
  return z

class Patch(nn.Module):
  def __init__(self, patch_size=4, in_channels=1, out_channels=32):
    super().__init__()
    self.out_channels = out_channels
    self.patch_size = patch_size
    self.conv = nn.Conv2d(in_channels, out_channels, 5, padding=2, stride=patch_size)

  def forward(self, x):
    """
    batch x c x h x w -> batch x (h//ps)*(w//ps) x d
    """
    b, c, h, w = x.shape
    x = self.conv(x)
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(b, h//self.patch_size * w//self.patch_size, self.out_channels)
    return x

class UnPatch(nn.Module):
  def __init__(self, patch_size=4, in_channels=32, out_channels=1):
    super().__init__()
    self.out_channels = out_channels
    self.patch_size = patch_size
    self.up = nn.Linear(in_channels, patch_size**2 * out_channels)

  def forward(self, x):
    """
     batch x (h//ps)*(w//ps) x d -> batch x c x h x w
    """
    b, s, d = x.shape
    x = self.up(x)
    w = int(s**0.5)
    h = w
    x = x.reshape(b, h, w, self.out_channels, self.patch_size, self.patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, self.out_channels, h*self.patch_size, w*self.patch_size)
    return x

class RMSNorm(nn.Module):
  def __init__(self, d=32, eps=1e-6):
    super().__init__()
    self.scale = nn.Parameter(t.ones((1,1,d)))
    self.eps = eps

  def forward(self, x):
    x = x/(((x**2).mean())**0.5 + self.eps)
    x = x * self.scale
    return x

class MLP(nn.Module):
  def __init__(self, d=32, exp=2):
    super().__init__()
    self.d = d
    self.exp = exp
    self.up = nn.Linear(d, exp*d, bias=False)
    self.gate = nn.Linear(d, exp*d, bias=False)
    self.down = nn.Linear(exp*d, d, bias=False)
    self.act = nn.SiLU()

  def forward(self, x):
    x = self.up(x) * self.act(self.gate(x))
    x = self.down(x)
    return x

class Attention(nn.Module):
  def __init__(self, d=32, n_head=4):
    super().__init__()
    self.n_head = n_head
    self.d = d
    self.d_head = d//n_head
    self.QKV = nn.Linear(d, 3*d, bias=False)
    self.O = nn.Linear(d, d, bias=False)
    self.normq = RMSNorm(self.d_head)
    self.normk = RMSNorm(self.d_head)

  def forward(self, x):
    b, s, d = x.shape
    q, k, v = self.QKV(x).chunk(3, dim=-1)
    q = q.reshape(b, s, self.n_head, self.d_head)
    k = k.reshape(b, s, self.n_head, self.d_head)
    v = v.reshape(b, s, self.n_head, self.d_head)
    q = self.normq(q)
    k = self.normk(k)
    attn = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1)
    attn = attn.softmax(dim=-1)
    # b, nh, sq, sk
    z = attn @ v.permute(0, 2, 1, 3)
    # b, nh, sq, dh
    z = z.permute(0, 2, 1, 3)
    z = z.reshape(b, s, self.d)
    z = self.O(z)
    return z

class NumEmbedding(nn.Module):
  def __init__(self, n_max, d=32, C=500):
    super().__init__()
    thetas = C**(-t.arange(0, d//2)/(d//2))
    thetas = t.arange(0, n_max)[:, None].float() @ thetas[None, :]
    sins = t.sin(thetas)
    coss = t.cos(thetas)
    self.register_buffer("E", t.cat([sins, coss], dim=1))

  def forward(self, x):
    return self.E[x]

class DiTBlock(nn.Module):
  def __init__(self, d=32, n_head=4, exp=2):
    super().__init__()
    self.d = d
    self.n_head = n_head
    self.exp = exp
    self.norm1 = RMSNorm(d)
    self.attn = Attention(d, n_head)
    self.norm2 = RMSNorm(d)
    self.mlp = MLP(d)
    self.modulate = nn.Linear(d, 6*d)

  def forward(self, x, c):
    """
    x ... b, s, d
    c ... b, d
    """
    scale1, bias1, gate1, scale2, bias2, gate2 = self.modulate(c).chunk(6, dim=-1)
    residual = x
    x = self.norm1(x) * (1+scale1[:, None, :]) + bias1[:, None, :]
    x = self.attn(x) * gate1[:, None, :]
    x = residual + x

    residual = x
    x = self.norm2(x) * (1+scale2[:, None, :]) + bias2[:, None, :]
    x = self.mlp(x) * gate2[:, None, :]
    x = residual + x
    return x


class DiT(nn.Module):
  def __init__(self, h, w, n_classes=10, in_channels=1, patch_size=4, n_blocks=4, d=32, n_head=4, exp=2, T=1000):
    super().__init__()
    self.T = T
    self.patch = Patch(patch_size, in_channels, d)
    self.n_seq = h//patch_size * w//patch_size
    self.pe = nn.Parameter(t.randn(1, self.n_seq, d)*d**(-0.5))
    self.te = NumEmbedding(T, d)
    self.ce = nn.Embedding(n_classes, d)
    self.act = nn.SiLU()
    self.blocks = nn.ModuleList([DiTBlock(d, n_head, exp) for _ in range(n_blocks)])
    self.norm = RMSNorm(d)
    self.modulate = nn.Linear(d, 2*d)
    self.unpatch = UnPatch(patch_size, d, in_channels)

  def forward(self, x, c, ts):
    """
    x ... b, c, h, w
    c ... b
    ts ... b
    """
    ts_int = t.minimum((ts * self.T).to(t.int64), t.tensor(self.T-1))
    cond = self.act(self.te(ts_int) + self.ce(c)) # b x d
    x = self.patch(x) + self.pe
    for idx, block in enumerate(self.blocks):
      x = block(x, cond)
    scale, bias = self.modulate(cond).chunk(2, dim=-1)
    x = self.norm(x) * (1+scale[:, None, :]) + bias[:, None, :]
    x = self.unpatch(x)
    return x
  
  @property 
  def dtype(self):
    return self.pe.dtype
  
  @property 
  def device(self):
    return self.pe.device



