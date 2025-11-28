import torch

class RotaryEmbedding():
    def __init__(
        self,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        dtype: torch.dtype,
    ) -> None:
        """This class implements rotary embeddings."""
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

    def pre_compute_freqs(self, dim, base) -> torch.Tensor:
        """Pre-compute the inverse frequencies for rotary embeddings."""
        # for example if rotary_dim=8, base=10000, the inv_freqs have the shape (4,)
        # then inv_freqs = [1/10000^(0/8), 1/10000^(2/8), 1/10000^(4/8), 1/10000^(6/8)]
        inv_freqs = 1.0 / base**(torch.arange(0, dim, 2, dtype=self.dtype) / dim)
        return inv_freqs

    def compute_cos_sin_cache(self):
        inv_freq = self.pre_compute_freqs(self.rotary_dim, self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        # Compute the outer product
        # For example, if t has shape (L,) and inv_freq has shape (D/2,),
        # t = [0, 1, 2, ..., L-1]
        # inv_freqs = [f1, f2, ..., fD/2]
        # and then: the freqs has the shape (L, D/2)
        # [0*f1, 0*f2, ..., 0*fD/2]
        # ...
        # [(L-1)*f1, (L-1)*f2, ..., (L-1)*fD/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
        
    def get_cos_sin(self, seqlen: int) -> tuple[torch.Tensor, torch.Tensor]:
        # cos: [seqlen, head_dim/2]
        # sin: [seqlen, head_dim/2]
        cos_sin = self.compute_cos_sin_cache()[:seqlen]
        cos, sin = cos_sin.chunk(2, dim=-1)
        return cos, sin

if __name__ == '__main__':
    rotary_emb = RotaryEmbedding(
        rotary_dim=8,
        max_position_embeddings=16,
        base=10000,
        dtype=torch.float32,
    )
    cos, sin = rotary_emb.get_cos_sin(seqlen=16)
    print("Cosine values:\n", cos)
    print(f"Cosine shape: {cos.shape}")
    print("Sine values:\n", sin)
    print(f"Sine shape: {sin.shape}")
