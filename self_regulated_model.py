import torch
from transformers import AutoModelForCausalLM


class PromptEngineeredModel:
    """
    Wrapper class for a AutoModelForCausalLM model that adds a system prompt to the input sequence.
    """

    def __init__(self, model, tokenizer, system_prompt):
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompt = self.tokenizer(
            system_prompt, return_tensors="pt"
        ).input_ids

    def forward(self, input_ids, **kwargs):
        if self.system_prompt is None:
            raise ValueError("System prompt has not been set.")

        # Concatenate system prompt with the input sequence
        input_ids = torch.cat((self.system_prompt, input_ids), dim=-1)

        # Pass the concatenated sequence to the model
        return self.model(input_ids, **kwargs)

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
