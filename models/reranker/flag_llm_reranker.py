import os
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class FlagLLMReranker(nn.Module):
    def __init__(
            self,
            model_dir: str = None,
            device: str = 'cpu') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.device = torch.device(device)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)[
            "input_ids"][0]
        self.initial_prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
        self.sep = "\n"

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 16,
                      max_length: int = 512, prompt: str = None, normalize: bool = True) -> List[float]:
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        prompt = prompt or self.initial_prompt

        for batch_start in trange(0, len(sentence_pairs), batch_size):
            batch_sentences = sentence_pairs[batch_start:batch_start + batch_size]
            batch_sentences = [
                f"A: {q}" + self.sep + f"A: {p}" + self.sep + prompt for q, p in batch_sentences]

            batch_inputs = self.tokenizer(text=batch_sentences, add_special_tokens=True,
                                          truncation=True, padding=True, max_length=max_length,
                                          return_attention_mask=True, return_tensors="pt")
            batch_inputs = batch_inputs.to(self.device)

            outputs = self.model(**batch_inputs, output_hidden_states=True)
            logits = outputs.logits
            scores = self.__last_logit_pool(
                logits, batch_inputs["attention_mask"])
            scores = scores[:, self.yes_loc]
            all_scores.extend(scores.cpu().float().tolist())

        if normalize:
            all_scores = [self.__sigmoid(score) for score in all_scores]
        return all_scores

    def __last_logit_pool(self, logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return logits[:, -1, :]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = logits.shape[0]
            return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
