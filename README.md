Generate piano accompaniment based on the principles of masked diffusion models (MDMs). Code and training paradigms based on the recently published ["Scaling up Masked Diffusion Models on Text"](https://arxiv.org/abs/2410.18514) and ["Large Language Diffusion Models"](https://arxiv.org/abs/2502.09992).

1. preprocess the MIDI data and train a tokenizer with BPE
2. pre-train a model from scratch.
3. Use supervised fine-tuning (SFT) to train the model to a specific task. (piano accompaniment)

- Use any existing MIDI dataset(s) to train a model. Modify the tokenizer settings to meet your needs. Currently TSD tokenizer is used.
