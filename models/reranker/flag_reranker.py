
import os
import warnings
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm, trange
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          is_torch_npu_available)

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class collater():
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_to_multiple_of = 8
        self.label_pad_token_id = -100
        warnings.filterwarnings("ignore",
                                message="`max_length` is ignored when `padding`=`True` and there is no truncation strategy.")

    def __call__(self, data):
        labels = [feature["labels"] for feature in data] if "labels" in data[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won"t pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in data:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        return self.tokenizer.pad(
            data,
            padding=True,
            max_length=self.max_len,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

def last_logit_pool(logits: Tensor,
                    attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1, :]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return torch.stack([logits[i, sequence_lengths[i], :] for i in range(batch_size)], dim=0)

def last_logit_pool_layerwise(logits: Tensor,
                              attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return logits[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = logits.shape[0]
        return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FlagReranker:
    def __init__(
            self,
            model_name_or_path: str = None,
            use_fp16: bool = False,
            cache_dir: str = None,
            device: Union[str, int] = None) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        if device and isinstance(device, str):
            self.device = torch.device(device)
            if device == "cpu":
                use_fp16 = False
        else:
            if torch.cuda.is_available():
                if device is not None:
                    self.device = torch.device(f"cuda:{device}")
                else:
                    self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif is_torch_npu_available():
                self.device = torch.device("npu")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        if use_fp16:
            self.model.half()

        self.model = self.model.to(self.device)

        self.model.eval()

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"----------using {self.num_gpus}*GPUs----------")
                self.model = torch.nn.DataParallel(self.model)
        else:
            self.num_gpus = 1

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 256,
                      max_length: int = 512, normalize: bool = False) -> List[float]:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            inputs = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(self.device)

            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            all_scores.extend(scores.cpu().numpy().tolist())

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores


class FlagLLMReranker:
    def __init__(
            self,
            model_dir: str = None) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
        self.initial_prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."

    @torch.no_grad()
    def compute_score(self, sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]], batch_size: int = 16,
                      max_length: int = 512, prompt: str = None, normalize: bool = True) -> List[float]:
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        length_sorted_idx = np.argsort([-self._text_length(q) - self._text_length(p) for q, p in sentence_pairs])
        sentences_sorted = [sentence_pairs[idx] for idx in length_sorted_idx]
        
        all_scores = []
        prompt = prompt or self.initial_prompt
        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)["input_ids"]
        sep = "\n"
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)["input_ids"]
        encode_max_length = max_length + len(sep_inputs) + len(prompt_inputs)
        
        for batch_start in trange(0, len(sentences_sorted), batch_size):
            batch_sentences = sentences_sorted[batch_start:batch_start + batch_size]
            queries = [f"A: {q}" for q, _ in batch_sentences]
            passages = [f"B: {p}" for _, p in batch_sentences]
            
            queries_inputs = self.tokenizer(queries,
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length * 3 // 4,
                                            truncation=True)["input_ids"]
            passages_inputs = self.tokenizer(passages,
                                            return_tensors=None,
                                            add_special_tokens=False,
                                            max_length=max_length,
                                            truncation=True)["input_ids"]

            batch_inputs = []
            for query_inputs, passage_inputs in zip(queries_inputs, passages_inputs):
                item = self.tokenizer.prepare_for_model(
                    [self.tokenizer.bos_token_id] + query_inputs,
                    sep_inputs + passage_inputs,
                    truncation="only_second",
                    max_length=encode_max_length,
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False
                )
                item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
                item["attention_mask"] = [1] * len(item["input_ids"])
                item.pop("token_type_ids") if "token_type_ids" in item.keys() else None
                if "position_ids" in item.keys():
                    item["position_ids"] = list(range(len(item["input_ids"])))
                batch_inputs.append(item)

            collater_instance = collater(self.tokenizer, max_length)
            batch_inputs = collater_instance(
                [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in
                    batch_inputs])

            batch_inputs = {key: val.to(self.device) for key, val in batch_inputs.items()}

            outputs = self.model(**batch_inputs, output_hidden_states=True)
            logits = outputs.logits
            scores = last_logit_pool(logits, batch_inputs["attention_mask"])
            scores = scores[:, self.yes_loc]
            all_scores.extend(scores.cpu().float().tolist())

        all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores


    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings


if __name__=='__main__':
    model_path = "/Users/miles/bge-reranker-v2-gemma"
    reranker = FlagLLMReranker(model_name_or_path=model_path, use_fp16=True)
    score = reranker.compute_score(["query", "passage"])
    print(score) # -5.65234375
    samples = [["what is panda?", "hi"], 
        ["what is panda?", "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."]]
    scores = reranker.compute_score(samples)
    print(scores)