# NN_Zero-to-Hero 🤖

*This repo is focused on a course by Andrej Karpathy on building neural networks, from scratch, in code. We start with the basics of backpropagation and build up to modern deep neural networks, like GPT.*

[Zero to Hero Playlist Link](https://youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&si=xZk-Ar3vdQ7i2K-z)

---
# Video 1: [The Spelled-Out Intro to Neural Networks and Backpropagation: Building Micrograd](https://youtu.be/VMj-3S1tku0?si=G6dX1T-LRp6ugqNA)
> This is the most step-by-step spelled-out explanation of backpropagation and training of neural networks. It only assumes basic knowledge of Python and a vague recollection of calculus from high school.
- 

# Video 2: [The Spelled-Out Intro to Language Modeling: Building Makemore](https://youtu.be/PaCmpygFfXo?si=JhZ6-_Otuv3w7qLn)
> We implement a bigram character-level language model, which we will further complexify in follow up videos into a modern Transformer language model, like GPT. In this video, the focus is on:
  1. Introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and 
  2. The overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).
- 

# Video 3: [Building Makemore Part 2: MLP](https://youtu.be/TCH_1BHY58I?si=MrKLOaL-xuQqF656)
> We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).
- 

# Video 4: [Building Makemore Part 3: Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc?si=wjTVEpTqJPZ4sRDq)
> We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization. Residual connections and the Adam optimizer remain notable to-dos for later video.
- 

# Video 5: [Building Makemore Part 4: Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI?si=CFwo9_y7fCyqqF_M)
> We take the 2-layer MLP (with BatchNorm) from the previous video and backpropagate through it manually without using PyTorch autograd's loss.backward(): through the cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer, and the embedding table. Along the way, we get a strong intuitive understanding about how gradients flow backwards through the compute graph and on the level of efficient Tensors, not just individual scalars like in micrograd. This helps build competence and intuition around how neural nets are optimized and sets you up to more confidently innovate on and debug modern neural networks.
- 

# Video 6: [Building Makemore Part 5: Building a WaveNet](https://youtu.be/t3YJ5hKiMQ0?si=SBsq5FHXzUau2FIL)
> We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between Jupyter notebooks and repository code, ...).
- 

# Video 7: [Let's Build GPT: From Scratch, In Code, Spelled Out.](https://youtu.be/kCc8FmEb1nY?si=eWfT6YZEtqBhcGph)
> We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.

# Video 8: [State of GPT | BRK216HFS](https://youtu.be/bZQun8Y4L2A?si=e3jnQYs0OoDM4Gz1)
> Learn about the training pipeline of GPT assistants like ChatGPT, from tokenization to pretraining, supervised finetuning, and Reinforcement Learning from Human Feedback (RLHF). Dive deeper into practical techniques and mental models for the effective use of these models, including prompting strategies, finetuning, the rapidly growing ecosystem of tools, and their future extensions.
- 
