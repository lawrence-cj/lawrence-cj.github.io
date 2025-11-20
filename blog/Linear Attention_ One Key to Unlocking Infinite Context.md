---
title: 'Linear Attention: One Key to Unlocking Infinite Context'

---

![Cover](https://hackmd.io/_uploads/HJWjwVnlbe.png)

# Linear Attention: One Key to Unlocking Infinite Context

In the age of AI dominated by Transformer models, the core component—the attention mechanism—is the undisputed star. However, the standard Softmax attention has a critical weakness: its computational and memory complexity grows quadratically ($O(N^2)$) with the sequence length $N$. This means that processing long texts, high-resolution images, and especially long videos, becomes prohibitively expensive.

To break through this bottleneck, researchers have proposed various efficient attention variants. Among them, **Linear Attention** stands out for its mathematical elegance and practical power. It successfully reduces the complexity to a linear scale ($O(N)$), opening the door to processing ultra-long sequences.

This post will dive deep into the core idea of Linear Attention, derive its magical **"state accumulation"** property, and demonstrate how it empowers LLMs and long video generation.

## The Evolution from Softmax to Linear Attention

Let's first recall the standard Softmax attention formula:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Here, $Q$, $K$, and $V$ are the Query, Key, and Value matrices. The bottleneck lies in the $QK^T$ matrix multiplication, which creates an $N \times N$ attention score matrix. This single step is responsible for the $O(N^2)$ complexity.

The core idea of Linear Attention is to **"aggregate first, query later,"** cleverly avoiding the creation of this massive $N \times N$ matrix. It achieves this by applying a feature map $\phi(\cdot)$ (such as ReLU or ELU+1) to the Queries and Keys and then leveraging the associative property of matrix multiplication to change the order of computation.

While standard attention computes $(QK^T)V$, linear attention reorders it to $Q(K^T V)$. Let's look at the formula:

$$
\text{Attention}(Q, K, V)_i = \frac{\phi(Q_i)^T \sum_{j=1}^{N} (\phi(K_j) V_j^T)}{\phi(Q_i)^T \sum_{j=1}^{N} \phi(K_j)}
$$

At first glance, this still seems to require iterating through all $j$ for each $i$. But the key insight is that the term $\sum_{j=1}^{N} (\phi(K_j) V_j^T)$ (a $d \times d$ matrix) can be pre-computed and **reused** for all queries $Q_i$. More importantly, in an autoregressive (causal) setting, this can be expressed as an **accumulating state**.

### I. The Core Magic: The State Accumulation Property

For autoregressive tasks like generative models, the output at position $i$ can only depend on inputs from positions $1$ to $i$. The linear attention formula thus becomes:

$$
O_i = \frac{\phi(Q_i)^T \sum_{j=1}^{i} (\phi(K_j) V_j^T)}{\phi(Q_i)^T \sum_{j=1}^{i} \phi(K_j)}
$$

Let's define a "state" $S_i$ that represents the accumulated Key-Value information up to step $i$:

$$
S_i = \sum_{j=1}^{i} \phi(K_j) V_j^T
$$

This state $S_i$ possesses a wonderfully simple recursive property:

$$
S_i = S_{i-1} + \phi(K_i) V_i^T
$$

This is the magic of Linear Attention! As illustrated in the image below, we don't need to recompute the sum from $j=1$ to $i$ at every step. We simply maintain a fixed-size state $S$ (along with a denominator term for normalization). At each new timestep $i$, we update the state by adding the current step's product $\phi(K_i) V_i^T$ to the previous state $S_{i-1}$.

![LINEAR ATTENTION MECHANISM](https://hackmd.io/_uploads/SJemHDmRxg.png)
*Figure 1: The computation in Linear Attention can be framed as a recursive process. To compute the output for block `i`, we only need the cached state from the previous `i-1` blocks ($S_{1 \rightarrow i-1}$) and the information from the current block ($K_i, V_i$).*

This "state accumulation" feature means that no matter how long the sequence gets, we only need to keep a fixed-size state matrix in memory. The computational complexity drops from $O(N^2)$ to $O(N)$, and the memory requirement plummets from $O(N^2)$ to $O(1)$ (with respect to sequence length).


---
## Long Context LLM via State Summation

### I. Mathematical Derivation of Retrieval Capability and Advanced Enhancement Mechanisms

While $O(N)$ complexity is achieved, vanilla Linear Attention encounters a strict mathematical bottleneck in accurate **Associative Retrieval** tasks. Next-generation models actively employ gating and error correction to overcome this limitation.

#### 1.1 Theoretical Root of Retrieval Failure: Orthogonality Limitation

The accumulated state $S$ is the sum of outer products of all Key $K_i$ and Value $V_i$: $S = \sum_{i=1}^{L} V_i K_i^T$.

**Definition and Expansion of the Retrieval Operation:**
The operation to retrieve $V_j$ using $K_j$ (as the query) is $\hat{V}_j = S K_j$. Substituting the definition of $S$:

$$\hat{V}_j = \left( \sum_{i=1}^{L} V_i K_i^T \right) K_j = \sum_{i=1}^{L} V_i (K_i^T K_j)$$

**Separation into Target Term and Interference Term:**
We separate the sum into the target term ($i=j$) and the interference term ($i \neq j$):

$$\hat{V}_j = \underbrace{V_j (K_j^T K_j)}_{\text{Target Term}} + \underbrace{\sum_{i \neq j} V_i (K_i^T K_j)}_{\text{Interference Term (Retrieval Error)}}$$

**Derivation of Perfect Retrieval Conditions:**
For perfect retrieval ($\hat{V}_j = V_j$), the **Interference Term must be zero**. This requires the coefficients of all $V_i$ to be zero:

$$K_i^T K_j = 0 \quad \text{must hold for all } i \neq j$$

This is the definition of **mutual orthogonality**. Therefore, the **maximum number of interference-free memories is strictly limited by the Key/Value dimension $d$** (Max $L \le d$). Once sequence length $L$ exceeds $d$, retrieval error is inevitable.

### 1.2 DeltaNet's Gating and Error Correction Mechanism

DeltaNet[1,2] introduces proactive error correction by treating state update as an online learning process that minimizes the **Mean Squared Error (MSE)**.

**Defining the Loss Function and Calculating the Gradient:**
At each step $t$, the state $S$ minimizes the MSE loss $L_t(S)$:
$$L_t(S) = \frac{1}{2} \| S K_t - V_t \|^2$$
We calculate the gradient of the loss with respect to $S$:
$$\nabla L_t(S) = \frac{\partial L_t(S)}{\partial S} = (S K_t - V_t) K_t^T$$

**Deriving DeltaNet's Recurrent Formula:**
Substituting the gradient into the gradient descent update rule $S_t = S_{t-1} - \beta_t \nabla L_t(S_{t-1})$ (setting $\eta_t = \beta_t$):

$$S_t = S_{t-1} - \beta_t (S_{t-1} K_t - V_t) K_t^T$$

**Formula Expansion and Mechanism Separation:**
Expanding and regrouping the terms, we derive DeltaNet's final recurrent formula:

$$S_t = \underbrace{S_{t-1} (I - \beta_t K_t K_t^T)}_{\text{Projection and Decay of Old State (Gating)}} + \underbrace{\beta_t V_t K_t^T}_{\text{Injection of New Information}}$$

**Mechanism Summary:** The matrix $(I - \beta_t K_t K_t^T)$ acts as a **dynamic projection matrix**. It utilizes the prediction error to correct the old state $S_{t-1}$, enabling proactive control and **elimination** of the interference component along the direction of $K_t$, thereby breaking the $L>d$ retrieval limitation.

---

## II. Architectural Refinement and Hardware Engineering: Overcoming $L>d$ Limits

The practical success of LLM implementation relies on translating these theoretical mechanisms into hardware-efficient implementations and adopting modern architectural best practices.

### 2.1 Architectural Optimization: Proactive Interference Elimination

* **Geometric Role of L2-Normalization:** Applying **L2-Normalization** to the Key vector $K$ imbues the state transition matrix $I - K K^T$ with a clear **geometric projection** property. It actively **erases** the component of $S$ that interferes with the current Key $K_t$, achieving efficient memory "cleanup."
* **Role of Short Convolution:** Adding small-kernel **Short Convolution** (Depthwise Separable Conv1D) to the $Q, K, V$ paths supplements the purely content-based linear attention with **Local Positional Bias**. This serves as a "shortcut" for forming **Induction Heads**, which is critical for enhancing the LLM's in-context learning capabilities.

### 2.2 Training Efficiency and Hybrid Architectures

* **Chunkwise Parallel Training:** To overcome the $O(L)$ sequential dependency during training, the **Chunkwise Parallel Algorithm** is employed. This method leverages **state checkpointing** and reformulates the recurrent calculations into GPU-friendly **Matrix Multiplication (MatMul)**, achieving hardware-efficient $O(N)$ time complexity for training.
* **Hybrid Architectures (Hybrid Models):** Given the theoretical retrieval ceiling of fixed-state models, mixing linear RNN layers with a **small number (e.g., 2-4 layers) of $O(N^2)$ Global Softmax Attention** layers is highly effective. This **Hybrid Architecture** maintains near-linear inference speed while utilizing the global attention's **unlimited memory capacity** (in theory) to **surpass pure Transformer models' performance** on complex retrieval tasks.


## Long Video Generation via State Summation

Another perfect example leveraging this theoretical elegance is Long video generation in diffusion world. A video consists of thousands of frames, making it an impossibly long sequence for traditional attention mechanisms. [SANA-Video[3]](https://nvlabs.github.io/Sana/Video/) is a project exploring in this field.

By ultiziling the state accumulation property of linear attention, we can implement an efficient long video generation model with the following steps:

1.  **Model Adaptation**: Take a pre-trained video model with linear attention layers or replace its Softmax attention layers with **Causal Linear Attention**. Then, fine-tune the model on video data.
2.  **Block-wise Processing and State Caching**: During inference, we can generate the video in smaller chunks or blocks.
    * After processing the first chunk (Block 1), we compute and save its final accumulated state, $S_1$.
    * To generate the second chunk (Block 2), we no longer need the original K and V values from all the frames in Block 1. We simply load the cached state $S_1$ and continue accumulating the Key-Value information from Block 2 on top of it to get $S_2$.
    * This process continues, with the state $S$ acting like a compressed memory, constantly absorbing new video information while its size remains constant.

[![Demo Video of Causal-Attention](https://hackmd.io/_uploads/By4nSE3gZe.jpg)](https://www.youtube.com/watch?v=-vuCn_d9Qjk)
*Figure 2: In Block Causal Linear Attention, the computation for the current block (e.g., Block_n) directly uses the cumulative sum ("Cum Sum") from all preceding blocks, eliminating the need to access the original KV cache. This provides the dual advantages of **Constant memory usage** and **Global Context** awareness. **Click to play.***

With this method, even when the model is generating the n-th block of the video, its attention calculation implicitly incorporates all historical information from the very first frame. This enables the model to maintain long-term temporal consistency, generating videos that are coherent and logically sound, without "forgetting" what happened at the beginning.


## Future: From Simple Accumulation to Advanced State Updates

However, the above simple linear accumulation is just the beginning. In recent years, researchers have discovered that designing more sophisticated online update rules can significantly enhance model performance and expressive power. These methods transform the state update itself into a learnable, dynamic process.

The table below shows how several advanced models have proposed their own unique state update mechanisms:

![Advanced State Updates](https://hackmd.io/_uploads/rJCGn_XRge.jpg)
*Table from: Griffin: On the Design of Recurrent Models for Efficient Language Modeling (https://arxiv.org/pdf/2412.06464)*

From this table, we can see a clear progression:
* **Mamba2** introduces a decay factor $\alpha_t$, allowing the state to selectively "forget" past information.
* **Longhorn** and **DeltaNet** introduce gate-like mechanisms (e.g., $\mathbf{I} - \epsilon_t \mathbf{k}_t \mathbf{k}_t^T$), which make the update dependent on the current input $\mathbf{k}_t$, dynamically adjusting the influence of the previous state $\mathbf{S}_{t-1}$.
* **Gated DeltaNet** further combines decay and gating to create an even more powerful update paradigm.

These advanced update rules allow models to achieve far greater modeling capacity than simple linear accumulation, all while maintaining linear complexity. The "state" evolves from a simple "memory accumulator" into a complex, dynamic memory unit.


### Conclusion

Linear Attention represents a fundamental shift from the brute-force quadratic approach of Softmax attention. Its core strength lies in a simple mathematical reordering that enables a powerful **state accumulation** property ($S_i = S_{i-1} + \phi(K_i) V_i^T$).

This elegant trick unlocks the ability to process sequences of virtually unlimited length with constant memory and linear time complexity. As we push the boundaries of AI, efficient methods like Linear Attention are paving the way for more scalable and powerful models across domains, from natural language to the final frontier of long-form video.


### Reference
[1] Yang, Songlin, et al. "Parallelizing linear transformers with the delta rule over sequence length." Advances in neural information processing systems 37 (2024): 115491-115522.
[2] https://sustcsonglin.github.io/blog/2024/deltanet-2/
[3] Chen, Junsong, et al. "Sana-video: Efficient video generation with block linear diffusion transformer." arXiv preprint arXiv:2509.24695 (2025).
