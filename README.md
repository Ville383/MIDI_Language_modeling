Generating MIDI data based on the principles of masked diffusion models (MDMs). Training paradigm based on the recently published large language diffusion model [LLaDA](https://arxiv.org/abs/2502.09992).

1. pre-train a model from scratch.
2. Use supervised fine-tuning (SFT) to train the model to a specific task.
3. (Reinforcment learning.)

You can use any existing MIDI dataset(s) to train your model and change the tokenizer settings to meet your needs. Currently REMI tokenizer is used.
