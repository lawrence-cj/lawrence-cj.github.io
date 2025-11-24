![Cover](https://hackmd.io/_uploads/HJWjwVnlbe.png)

# Infinite Context Length with Global but Constant Attention Memory

In the age of AI dominated by Transformer models, the core component—the attention mechanism—is the undisputed star. The $O(N^2)$ complexity bottleneck in Transformer models poses a formidable hardware challenge. In practice, the linearly growing KV Cache of standard Softmax Attention is rapidly consuming the scarce and VERY EXPENSIVE ($$$) DRAM (Working Memory)[1]. This memory consumption model imposes a critical cost and capacity limit on the scalability of long-context models.

![image](https://hackmd.io/_uploads/rJm4FieZZx.png)

To solve this, Linear Attention provides an elegant mathematical solution. It not only reduces computational complexity to $O(N)$, but, more critically, it achieves a fixed-size $O(1)$ state.

This post explores the core mechanism of Linear Attention, deriving its state accumulation property and demonstrating how its constant memory bypasses the DRAM bottleneck, laying the foundation for economical and efficient ultra-long sequence generation.

![storage](https://hackmd.io/_uploads/Hy6W9ie--g.png)
*Figure 2: DRAM (left) acts as ultra-fast but expensive working memory for "hot data" like the active KV cache, frequently becoming a capacity bottleneck. NAND flash (right) provides economical, massive bulk storage for "cold data" such as datasets and model weights.*

## The Evolution from Softmax to Linear Attention

### I. The Difference in Mechanism: Softmax vs. Linear Attention

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

![image](https://hackmd.io/_uploads/SJhemvlZbx.png)
*Figure 3: Vanilla Attention (left) requires computing a large $N \times N$ attention matrix (green). Linear Attention (right) leverages matrix associativity to instead compute a compact, fixed-size $d \times d$ state matrix ($\sum K_i V_i$), resulting in linear complexity with respect to the sequence length $N$.*


### II. The Difference in KV Cache: Softmax vs. Linear Attention

In **Autoregressive Generation** tasks, the model must produce a new token at each step, relying on all historical tokens (the context) to compute attention. To avoid recomputing Keys and Values for the entire historical sequence at every step, the KV Cache mechanism becomes an essential choice for handling long contexts.

However, the two attention mechanisms utilize fundamentally different KV Cache implementations, directly determining their capacity to handle ultra-long sequences.

![kvcache-compare](https://hackmd.io/_uploads/S1HZ_Pgb-e.jpg)
*Figure 4: During a new query operation, Softmax attention (left) must attend to a growing KV cache representing the entire sequence history. In contrast, Linear Attention (right) utilizes a fixed-size state matrix ($S_{t-1}$), updated recursively with the new key-value information ($K_t V_t$), resulting in constant-time computation. (Nano-Banana 2 generated).*

#### Softmax Attention: Linearly Growing KV Cache
For standard Softmax Attention, when generating the output $O_t$ at position $t$, the Query $Q_t$ must compute the dot product against all past Keys $K_{1:t}$ to produce the weights:$$\text{Attention}(Q_t, K_{1:t}, V_{1:t}) = \text{softmax}\left(\frac{Q_t K_{1:t}^T}{\sqrt{d_k}}\right) V_{1:t}$$The KV Cache implementation, as shown on the **left side of Figure 4**, requires storing the entire history of Keys $K_{1:t}$ and Values $V_{1:t}$. With every step of generation, a new $K_t$ and $V_t$ must be appended to the Cache. Consequently, the KV Cache's memory size grows linearly with the sequence length $N$ (or the current timestep $t$), resulting in an $O(N)$ memory complexity. This massive, growing KV Cache quickly exhausts memory in ultra-long context settings, representing a major computational bottleneck as the Query $Q_t$ must be computed against the entire expanding sequence.


#### Linear Attention: Fixed-Size State $S$

Linear Attention completely avoids the need to store and re-access all historical Keys and Values through mathematical reordering:

$$O_i = \frac{\phi(Q_i)^T \sum_{j=1}^{i} (\phi(K_j) V_j^T)}{\phi(Q_i)^T \sum_{j=1}^{i} \phi(K_j)}$$

As illustrated on the **right side of Figure 4**, the core of Linear Attention lies in defining a fixed-size hidden state $S_i$ that compresses all historical Key-Value information into a single matrix. This state $S_i$ is defined as the accumulated sum of the Key and Value outer products:

$$S_i = \sum_{j=1}^{i} \phi(K_j) V_j^T$$

As seen in **Figure 5**, the new state $S_i$ is obtained recursively, using only the previous state $S_{i-1}$ and the current input $K_i, V_i$:

$$S_i = \sum_{j=1}^{i-1} \phi(K_j) V_j^T + \phi(K_i) V_i^T = S_{i-1} + \phi(K_i) V_i^T$$

The implementation achieves constant memory because the model does not need to store $K_{1:t}$ and $V_{1:t}$. Instead, it only needs to store and update a single, fixed-size matrix $S$ (with dimension $d \times d$). The memory complexity is strictly $O(1)$ (with respect to sequence length $N$), as the size of $S$ depends exclusively on the model's hidden dimension $d$. This guarantees that no matter how long the sequence becomes, the memory footprint remains constant, offering highly efficient inference where the Query $Q_t$ only multiplies the fixed state $S$.

![LINEAR ATTENTION MECHANISM](https://hackmd.io/_uploads/HJ0Ulhlbbl.png)
*Figure 5: The computation in Linear Attention can be framed as a recursive process. To compute the output for block `i`, we only need the cached state from the previous `i-1` blocks ($S_{1 \rightarrow i-1}$) and the information from the current block ($K_i, V_i$).*

---

## Long Video Generation via Causal Linear Attention

[SANA-Video[2]](https://nvlabs.github.io/Sana/Video/) leverages this theoretical elegance in long video generation, where a generated video consists of thousands of frames, making it an impossibly long sequence for traditional attention mechanisms. 

As shown in the **Figure 6**, by ultiziling the state accumulation property of linear attention, we can implement an efficient long video generation model with the following steps:

1. **Model Adaptation Enabling Causality via Fine-tuning:** We take a pre-trained video model already utilizing Linear Attention layers and fine-tune it specifically to operate as Causal Linear Attention on video data. This adaptation ensures that for any given frame, the state accumulation $S_i$ is strictly dependent only on preceding frames (from $j=1$ to $i$), which is a necessary condition for autoregressive generation.
2.  **Block-wise Processing and State Caching**: During inference, we can generate the video in smaller chunks or blocks.
    * After processing the first chunk (Block 1), we compute and save its final accumulated state, $S_1$.
    * To generate the second chunk (Block 2), we no longer need the original K and V values from all the frames in Block 1. We simply load the cached state $S_1$ and continue accumulating the Key-Value information from Block 2 on top of it to get $S_2$.
    * This process continues, with the state $S$ acting like a compressed memory, constantly absorbing new video information while its size remains constant.

[![Demo Video of Causal-Attention](https://hackmd.io/_uploads/By4nSE3gZe.jpg)](https://www.youtube.com/watch?v=-vuCn_d9Qjk)
*Figure 6: In Block Causal Linear Attention, the computation for the current block (e.g., Block_n) directly uses the cumulative sum ("Cum Sum") from all preceding blocks, eliminating the need to access the original KV cache. This provides the dual advantages of **Constant memory usage** and **Global Context** awareness. **Click to play.***

With this method, even when the model is generating the n-th block of the video, its attention calculation implicitly incorporates all historical information from the very first frame. This enables the model to maintain long-term temporal consistency, generating videos that are coherent and logically sound, without "forgetting" what happened at the beginning.


## Future: Advanced State Updating Strategy

However, the above simple linear accumulation is just the beginning. In recent years, researchers have discovered that designing more sophisticated online update rules can significantly enhance model performance and expressive power in large language models (LLMs). These methods transform the state update itself into a learnable, dynamic process.

The table below shows how several advanced models have proposed their own unique state update mechanisms:

![Advanced State Updating Strategy](https://hackmd.io/_uploads/rJCGn_XRge.jpg)
*Table 1: On the Design of Recurrent Models for Efficient Language Modeling[5]*

From this table, we can see a clear progression:
* **Mamba2** introduces a decay factor $\alpha_t$, allowing the state to selectively "forget" past information.
* **Longhorn** and **DeltaNet** introduce gate-like mechanisms (e.g., $\mathbf{I} - \epsilon_t \mathbf{k}_t \mathbf{k}_t^T$), which make the update dependent on the current input $\mathbf{k}_t$, dynamically adjusting the influence of the previous state $\mathbf{S}_{t-1}$.
* **Gated DeltaNet** further combines decay and gating to create an even more powerful update paradigm.

These advanced update rules allow models to achieve far greater modeling capacity than simple linear accumulation, all while maintaining linear complexity. The "state" evolves from a simple "memory accumulator" into a complex, dynamic memory unit.

---
## Background: Long Context LLM via Causal Linear Attention

In this section, we introduce how state updating strategies are motivated and designed in LLMs.

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

DeltaNet[3,4] introduces proactive error correction by treating state update as an online learning process that minimizes the **Mean Squared Error (MSE)**.

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



### Conclusion

Linear Attention represents a pivotal shift from the prohibitive $O(N^2)$ approach to a scalable, hardware-conscious paradigm. By mathematically reordering the attention calculation, it enables the fundamental **state accumulation property** ($S_i = S_{i-1} + \phi(K_i) V_i^T$) that collapses memory usage from linear to constant **$O(1)$**, directly bypassing the costly DRAM bottleneck.

Furthermore, advanced strategies like correction and gated recurrences successfully enhance this fixed state's modeling capacity, solving the crucial associative retrieval problem while preserving linear complexity. This elegant combination of constant memory efficiency and dynamic state capacity is the key to **unlocking truly scalable and powerful generative models across long-form video, LLMs, and beyond.**

### Reference

[1] Han, Song, et al. "Learning both weights and connections for efficient neural network." Advances in neural information processing systems 28 (2015).
[2] Chen, Junsong, et al. "Sana-video: Efficient video generation with block linear diffusion transformer." arXiv preprint arXiv:2509.24695 (2025).
[3] Yang, Songlin, et al. "Parallelizing linear transformers with the delta rule over sequence length." Advances in neural information processing systems 37 (2024): 115491-115522.
[4] https://sustcsonglin.github.io/blog/2024/deltanet-2/
[5] De, Soham, et al. "Griffin: Mixing gated linear recurrences with local attention for efficient language models." arXiv preprint arXiv:2402.19427 (2024).
[6] Dao, Tri, and Albert Gu. "Transformers are ssms: Generalized models and efficient algorithms through structured state space duality." arXiv preprint arXiv:2405.21060 (2024).
[7] Yang, Songlin, Jan Kautz, and Ali Hatamizadeh. "Gated delta networks: Improving mamba2 with delta rule." arXiv preprint arXiv:2412.06464 (2024).
[8] Liu, Bo, et al. "Longhorn: State space models are amortized online learners." arXiv preprint arXiv:2407.14207 (2024).
