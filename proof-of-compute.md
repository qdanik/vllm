# Proof of Compute (PoC) for LLM Nodes

## Overview

Proof of Compute is a mechanism to verify that nodes have sufficient memory bandwidth and capacity to run LLM inference. Nodes compete by searching for valid nonces during a time window, proving their hardware capabilities.

---

## PoC Versions

| Version | Model Weights | Use Case |
|---------|---------------|----------|
| **PoC 1.0** | Random init (from seed) | Standalone benchmark |
| **PoC 2.0** | Real model + shuffle/random layers | Production inference nodes |

---

## How PoC Works

### Round Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. INIT MODEL                                              │
│     - Load base model weights                               │
│     - Apply deterministic shuffle (from block seed)         │
├─────────────────────────────────────────────────────────────┤
│  2. NONCE SEARCH (5 min window)                             │
│     ┌─────────────────────────────────────────────────────┐ │
│     │  for nonce in range(∞):                             │ │
│     │      input = hash_to_vector(nonce)  # skip embed    │ │
│     │      output = model.forward(input)                  │ │
│     │      if meets_difficulty(output):                   │ │
│     │          valid_nonces.append(nonce)                 │ │
│     └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  3. SUBMIT                                                  │
│     - Submit valid nonces to chain                          │
│     - Validators re-run inference to verify                 │
│     - Reward ∝ valid nonces found                           │
└─────────────────────────────────────────────────────────────┘
```

### Step Details

| Step | Description |
|------|-------------|
| **Init Model** | Load model with shuffled weights (shuffle determined by block seed) |
| **Generate Input** | `nonce → hash → float vector` — bypasses embedding layer |
| **Inference** | Full forward pass through transformer layers |
| **Check Condition** | `hash(output) < difficulty_target` |
| **Collect Nonces** | Store valid nonces for submission |

---

## Security Model

### What We're Proving

Nodes must prove they have:
1. **Memory Capacity** — Can store full model weights
2. **Memory Bandwidth** — Can read weights fast enough for inference

### The Attack to Prevent

**Cheap GPU emulating expensive GPU:**
- Attacker has 8GB GPU
- Wants to run 28GB model
- Attempts to compute weights on-the-fly instead of storing them

---

## PoC 1.0: Random Weight Initialization

### How It Works

Weights are **randomly generated** from a deterministic seed at round start:

```python
# All weights generated from block seed
def init_model(block_seed: int, model_config):
    rng = torch.Generator().manual_seed(block_seed)
    for layer in model.layers:
        layer.weight = torch.randn(layer.shape, generator=rng)
```

### Limitations

| Approach | Problem |
|----------|---------|
| Random pool | Pool fits in L2 cache → fast lookup attack |
| Hash-based (`weight[i] = hash(seed, i)`) | Weights are "compressible" to seed → compute on-the-fly attack |

**Core issue:** Generated weights can always be recomputed — attacker trades compute for memory.

---

## PoC 2.0: Real Model + Randomization

### Motivation

Production nodes already run **real inference** using vLLM. We want:
- ✅ Seamless switch between inference and PoC
- ✅ Ability to run PoC **simultaneously** with inference
- ✅ No model reload overhead
- ✅ Leverage incompressibility of real trained weights

### Core Principle

> **Real trained weights are incompressible.**
> You either have them in memory, or you don't.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  vLLM Server (always running)                               │
│  └── Real 14B Model loaded in GPU memory                    │
├─────────────────────────────────────────────────────────────┤
│  Normal Mode: Serve inference requests                      │
├─────────────────────────────────────────────────────────────┤
│  PoC Mode: Apply randomization → run nonce search           │
│            └── Option A: Temporary weight shuffle           │
│            └── Option B: Additional random layers           │
└─────────────────────────────────────────────────────────────┘
```

### Option A: Temporary Weight Shuffle

Apply deterministic row permutation to weight matrices:

```python
# At PoC round start
rng = RandomGenerator(block_seed)
for layer in model.layers:
    perm = rng.permutation(layer.weight.shape[0])
    layer.weight = layer.weight[perm, :]  # Row shuffle

# After PoC round — restore original order
for layer in model.layers:
    layer.weight = layer.weight[inverse_perm, :]
```

**Trade-off:** Requires weight reorder (~seconds), blocks inference during PoC.

### Option B: Additional Random Layers (Preferred)

Inject lightweight random layers that don't modify base weights:

```python
class PoCWrapper(nn.Module):
    def __init__(self, base_model, block_seed):
        self.base_model = base_model  # Unchanged
        self.random_proj = RandomProjection(block_seed)  # Lightweight
    
    def forward(self, x):
        x = self.random_proj.pre(x)   # Random transform
        x = self.base_model(x)         # Real inference
        x = self.random_proj.post(x)  # Random transform
        return x
```

**Advantages:**
- Base model weights **unchanged** — can serve inference simultaneously
- Random layers are small (~MBs) but force full model execution
- Switch between modes is instant (no weight reordering)
- vLLM integration friendly

### Why This Works

| Property | Guarantee |
|----------|-----------|
| **Incompressible** | Real weights cannot be regenerated on-the-fly |
| **Unpredictable** | Random layers/shuffle depend on future block seed |
| **Non-disruptive** | Base model stays intact for production inference |
| **Bandwidth proof** | Must run full forward pass through real model |

---

## Implementation Details

### Input Generation (Skip Embedding)

Both versions bypass the embedding layer — input is generated directly:

```python
def nonce_to_input(nonce: int, seed: int, dim: int) -> Tensor:
    """Convert nonce to model input, bypassing embedding layer."""
    combined = hash64(seed ^ nonce)
    
    # Generate input vector deterministically
    rng = torch.Generator().manual_seed(combined)
    return torch.randn(1, seq_len, dim, generator=rng)
```

### Random Projection Layer (PoC 2.0)

```python
class RandomProjection(nn.Module):
    """Lightweight random layers for PoC 2.0."""
    
    def __init__(self, block_seed: int, dim: int):
        super().__init__()
        rng = torch.Generator().manual_seed(block_seed)
        
        # Small random matrices (~few MB total)
        self.pre_proj = torch.randn(dim, dim, generator=rng) * 0.01
        self.post_proj = torch.randn(dim, dim, generator=rng) * 0.01
    
    def pre(self, x: Tensor) -> Tensor:
        return x + x @ self.pre_proj  # Residual connection
    
    def post(self, x: Tensor) -> Tensor:
        return x + x @ self.post_proj
```

### Difficulty Check

```python
def check_nonce(output: Tensor, difficulty: int) -> bool:
    """Check if output meets difficulty target."""
    output_hash = hash256(output.cpu().numpy().tobytes())
    return int.from_bytes(output_hash[:8], 'big') < difficulty
```

---

## Properties

| Property | Guarantee |
|----------|-----------|
| **Determinism** | Same seed → same shuffle → same outputs |
| **Reproducibility** | Validators can verify any nonce |
| **Incompressibility** | Model weights cannot be computed on-the-fly |
| **Fairness** | Reward ∝ memory bandwidth (inference speed) |

---

## Parameters

| Parameter | Typical Value | Notes |
|-----------|---------------|-------|
| Model size | 14B params (~28GB) | Large enough to require HBM |
| Round duration | 5 minutes | Time window for nonce search |
| Sequence length | 256 | Input sequence for inference |
| Difficulty | Adjusted per round | Targets ~N valid nonces per round |

---

## Verification

Validators verify submitted nonces by:
1. Loading same model with same shuffle (from block seed)
2. Generating input from nonce
3. Running inference
4. Checking output meets difficulty

If output matches, nonce is valid. Reward is issued.

