# d_model:
hidden size of the model
larger d_model gives each token more representational capacity
this can capture richer semantic meaning, but costs more compute/memory

# num_heads:
number of parallel attention heads
more heads allow the model to learn more attention patterns at once
however, head_dim = d_model // num_heads
so increasing num_heads makes each head smaller
fewer heads = larger heads but fewer relationship patterns
more heads = more relationship patterns but smaller/weaker heads
common default: head_dim ≈ 64

# num_layers:
number of Transformer blocks
more layers give the model more stages to refine representations
deeper representations can become more abstract and context-aware
better representations can make predictions sharper / more separable
but more layers also increase compute, memory, and overfitting risk