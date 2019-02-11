## Neural Language Model

Tested on *PTB dataset* with perplexity: **116.114** using the below hyperparams.

Hyperparameters:
- Model: LSTM with 200 hidden units
- Batch size: 20
- Sequence length: 20
- Number of layers: 2
- Optimizer: Stochastic Gradient Descent
- Learning rate: initial value 0.1 with decay 0.5 starting from epoch 5
