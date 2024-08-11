import logging
import os
from typing import Dict, List

import torch
from torch import nn
from transformers import XLMRobertaModel, AutoTokenizer

logger = logging.getLogger(__name__)

class BGEM3Model(nn.Module):
    def __init__(self,
                 model_dir: str = None,
                 normlized: bool = True,
                 sentence_pooling_method: str = 'cls',
                 colbert_dim: int = -1):
        super().__init__()
        self.load_model(model_dir=model_dir, colbert_dim=colbert_dim)
        self.vocab_size: int = self.embedding.config.vocab_size

        self.normlized: bool = normlized
        self.sentence_pooling_method: str = sentence_pooling_method

    def load_model(self, model_dir: str, colbert_dim: int = -1):
        self.embedding: XLMRobertaModel = XLMRobertaModel.from_pretrained(model_dir)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.colbert_linear: nn.Linear = nn.Linear(in_features=self.embedding.config.hidden_size,
                                                         out_features=self.embedding.config.hidden_size if colbert_dim == -1 else colbert_dim)
        self.sparse_linear: nn.Linear = nn.Linear(
            in_features=self.embedding.config.hidden_size, out_features=1)

        if self.__check_pooler_existence(model_dir=model_dir):
            logger.info(
                'Loading existing colbert_linear and sparse_linear---------')
            self.load_pooler(model_dir=model_dir)
        else:
            logger.info(
                'The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')

    def __check_pooler_existence(self, model_dir: str):
        is_cobert_exist = os.path.exists(os.path.join(model_dir, 'colbert_linear.pt'))
        is_sparse_linear_exist = os.path.exists(os.path.join(model_dir, 'sparse_linear.pt'))
        return  is_cobert_exist and is_sparse_linear_exist
    
    def load_pooler(self, model_dir):
        colbert_state_dict = torch.load(os.path.join(
            model_dir, 'colbert_linear.pt'), map_location='cpu')
        sparse_state_dict = torch.load(os.path.join(
            model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)

    def dense_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d

    def sparse_embedding(self, hidden_state, input_ids, return_embedding: bool = True):
        token_weights = torch.relu(self.sparse_linear(hidden_state))
        if not return_embedding:
            return token_weights

        sparse_embedding = torch.zeros(input_ids.size(0), input_ids.size(1), self.vocab_size,
                                       dtype=token_weights.dtype,
                                       device=token_weights.device)
        sparse_embedding = torch.scatter(
            sparse_embedding, dim=-1, index=input_ids.unsqueeze(-1), src=token_weights)

        unused_tokens = [self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                         self.tokenizer.unk_token_id]
        sparse_embedding = torch.max(sparse_embedding, dim=1).values
        sparse_embedding[:, unused_tokens] *= 0.
        return sparse_embedding

    def colbert_embedding(self, last_hidden_state, mask):
        colbert_vecs = self.colbert_linear(last_hidden_state[:, 1:])
        colbert_vecs = colbert_vecs * mask[:, 1:][:, :, None].float()
        return colbert_vecs

    def dense_score(self, q_reps, p_reps):
        scores = BGEM3Model.compute_similarity(q_reps, p_reps)
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def sparse_score(self, q_reps, p_reps):
        scores = BGEM3Model.compute_similarity(q_reps, p_reps)
        scores = scores.view(q_reps.size(0), -1)
        return scores

    def colbert_score(self, q_reps, p_reps, q_mask: torch.Tensor):
        token_scores = torch.einsum('qin,pjn->qipj', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = scores.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)
        return scores

    def forward(self,
                text_input: Dict[str, torch.Tensor] = None,
                return_dense: bool = True,
                return_sparse: bool = False,
                return_colbert: bool = False,
                return_sparse_embedding: bool = False):
        assert return_dense or return_sparse or return_colbert, 'Must choose one or more from `return_colbert`, `return_sparse`, `return_dense` to set `True`!'

        with torch.no_grad():
            last_hidden_state = self.embedding(**text_input, return_dict=True).last_hidden_state

            output = {}
            if return_dense:
                dense_vecs = self.dense_embedding(last_hidden_state, text_input['attention_mask'])
                output['dense_vecs'] = dense_vecs
            if return_sparse:
                sparse_vecs = self.sparse_embedding(last_hidden_state, text_input['input_ids'],
                                                    return_embedding=return_sparse_embedding)
                output['sparse_vecs'] = sparse_vecs
            if return_colbert:
                colbert_vecs = self.colbert_embedding(last_hidden_state, text_input['attention_mask'])
                output['colbert_vecs'] = colbert_vecs

            if self.normlized:
                if 'dense_vecs' in output:
                    output['dense_vecs'] = nn.functional.normalize(output['dense_vecs'], dim=-1)
                if 'colbert_vecs' in output:
                    output['colbert_vecs'] = nn.functional.normalize(output['colbert_vecs'], dim=-1)

        return output


    def _tokenize(self, texts: list, max_length: int):
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            return_tensors='pt'
        )

    def tokenize(self, texts: List, max_length: int):
        return self._tokenize(texts=texts, max_length=max_length)