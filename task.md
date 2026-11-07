Design my chess AI project. Project: Build my own chess engine using reinforcement learning. Data: Self-play. Ambition: A serious AlphaZero-inspired project. Hardware: RTX 5050/5060 Laptop, 8GB VRAM, 32GB RAM, experienced Python developer. Give me exact architecture, code structure, training loop, datasets and realistic hardware settings.

build this as a small, complete AlphaZero-style system, rather than trying to reproduce DeepMind’s original training scale. Your hardware is enough to train a meaningful chess network and run self-play, but the limiting factor will likely be self-play throughput and MCTS, not just GPU memory.

The project will have five major parts:
1. A legal chess environment and action representation.
2. A neural network that predicts move probabilities and position value.
3. Monte Carlo Tree Search (MCTS) guided by that network.
4. Self-play games that generate training examples.
5. A training and evaluation pipeline that continually improves the network.
The key design decision: use MCTS to choose moves during self-play, and train the network to imitate MCTS visit distributions while predicting game outcomes. That is the central AlphaZero learning loop.


This is the AlphaZero learning loop: self-play creates examples, MCTS supplies improved move targets, and the network learns to predict those targets and the eventual game result. The original system used a much larger network and vastly more compute than your laptop, so we'll scale the architecture down while preserving the essential algorithm.