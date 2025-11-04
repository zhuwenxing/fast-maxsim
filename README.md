# fast-maxsim

> **Note**: This project is forked from [mixedbread-ai/maxsim-cpu](https://github.com/mixedbread-ai/maxsim-cpu). The main improvement is adding support for **macOS Intel** (x86_64) pre-built wheels, in addition to the original Linux x86_64 and macOS ARM support.

`fast-maxsim` is a high-performance CPU implementation of MaxSim scoring for late-interaction (ColBERT, ColPali) workflows.

It is a python library written in Rust and powered by `libsxmm` on x86 CPUs and Apple Accelerate on ARM macs. It supports Linux x86_64, macOS ARM (Apple Silicon), and macOS Intel (x86_64) platforms.

`fast-maxsim` is built to run exclusively on CPU, and achieves speed-ups that scale with core count on the scoring machine. It's designed to be used in situations where index/scoring machines do not have access to GPUs, and achieves ~2-3x speed-ups on ARM macs and 5x speedups on Linux CPUs over common PyTorch maxsim implementations.

It also implements effective just-in-time batching and padding for variable documents, greatly reducing padding overhead and needless computations.

## Getting Started

Pre-built wheels are available on Pypi for Python 3.9 through 3.13 and can be installed in the usual way:

```bash
uv pip install fast-maxsim # You may use vanilla pip install but why would you? If you're sophisticated, you could use `uv add` too!
```

Once installed, the simple API exposes two methods. For uniform-length inputs, you may use:

```python
import numpy as np
import fast_maxsim

# Prepare normalized embeddings
query = np.random.randn(32, 128).astype(np.float32)  # [num_query_tokens, dim]

# NOTE: fast-maxsim expects normalized vectors.
query /= np.linalg.norm(query, axis=1, keepdims=True)

docs = np.random.randn(1000, 512, 128).astype(np.float32)  # [num_docs, doc_len, dim]
# Normalize document embeddings...

# Compute MaxSim scores
scores = fast_maxsim.maxsim_scores(query, docs)  # Returns [num_docs] scores
```

For variable length inputs, you should use the alternate `maxsim_scores_variable`:

```python
import numpy as np
import fast_maxsim

# Prepare normalized embeddings
query = np.random.randn(32, 128).astype(np.float32)  # [num_query_tokens, dim]

# NOTE: fast-maxsim expects normalized vectors.
query /= np.linalg.norm(query, axis=1, keepdims=True)

# Create variable-length documents as a list
docs = [
    np.random.randn(np.random.randint(50, 800), 128).astype(np.float32)  # Variable length docs
    for _ in range(1000)
]
# Normalize document embeddings...

# Compute MaxSim scores
scores = fast_maxsim.maxsim_scores_variable(query, docs)  # Returns [num_docs] scores
```

## Platform Requirements

- **macOS ARM**: Apple Silicon (M1+)
- **macOS Intel**: x86_64 with AVX2 (Intel Haswell 2013+ - Core i3/i5/i7 4xxx series or newer)
- **Linux**: x86_64 with AVX2 (Intel Haswell 2013+, AMD Excavator 2015+)

We currently do not support Windows or take advantage of AVX512 instructions, nor do we optimise caching for specific CPUs. Contributions/PRs in this direction are welcome!

**Note**: Pre-built wheels on PyPI are available for Linux x86_64, macOS ARM (Apple Silicon), and macOS Intel (x86_64).

## Building

We use `maturin` as our build system. 

#### Linux

The easy way to build `fast-maxsim` from source on Linux is as follows:

```bash
# Install necessary system deps
apt-get install libssl-dev libopenblas-dev -y
apt-get install pkg-config -y
# Install tooling
uv pip install maturin patchelf numpy
# Install libxsmm
git@github.com:libxsmm/libxsmm.git && cd libxsmm && make STATIC=1 && make
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
# Clone and install fast-maxsim
git clone git@github.com:zhuwenxing/fast-maxsim.git
cd fast-maxsim
RUSTFLAGS="-L native=$(pwd)/../libxsmm/lib" maturin build --release --features use-libxsmm
```

Step by step:
- This installs OpenSSL and OpenBLAS, which will be required for compiling, as well as pkg-config so they can be found easily.
- It then clones `libxsmm`, on which most of the performance depends, and installs it.
- Installs RUST and enables its environment
- Clones this repository and finally build it

You may modify it and remove any step depending on dependencies already present on your machine.

#### Mac

On Mac, the installation is simplified, assuming you use homebrew:

**For Apple Silicon (M1+):**
```bash
# Install maturin
uv pip install maturin
# Install patchelf
brew install patchelf
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
# Clone and install fast-maxsim
git clone git@github.com:zhuwenxing/fast-maxsim.git
cd fast-maxsim
maturin build --release
```

**For Intel Mac (x86_64):**
```bash
# Install maturin
uv pip install maturin
# Install patchelf
brew install patchelf
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
. "$HOME/.cargo/env"
# Clone and install fast-maxsim
git clone git@github.com:zhuwenxing/fast-maxsim.git
cd fast-maxsim
# Build with AVX2 support (requires Intel Haswell 2013+ or newer)
RUSTFLAGS="-C target-cpu=haswell" maturin build --release
```

## Performance

For documents of uniform lengths, performance on Linux is slower than Jax on 4 core machines and either somewhat faster or slower depending on the CPU at 8 cores, and always faster than alternatives on ARM Macs. For variable document lengths (evaluated as a uniform distribution between 128 and 1536 tokens), `fast-maxsim` is always pretty fast thanks to more efficient batching.

### Mac M4 Ultra

![Mac M4 Ultra performance](speedup_comparisons/maxsim_speedup_mac.png)


### Linux AMD EPYC

#### 32 core limit performance

![Linux AMD EPYC 32 core performance](speedup_comparisons/maxsim_speedup_32cores.png)

#### 16 core limit performance

*It seems our performance was hindered during benchmarking due to a Rayon config issue when limiting the available cores. Leaving reporting as-is for now but performance is expected to be considerably better on an actual 16-core CPU.*

![Linux AMD EPYC 16 core performance](speedup_comparisons/maxsim_speedup_16cores.png)
