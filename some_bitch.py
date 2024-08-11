import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from models.bge_m3 import BGEM3Model

logger = logging.getLogger(__name__)

class BGEM3FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_fp16: bool = True,
            device: str = None) -> None:

        if device:
            self.num_gpus = 1
            self.device = torch.device(device)
        else:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"----------using {self.num_gpus}*GPUs----------")
                self.model.embedding = nn.DataParallel(self.model.embedding)
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                use_fp16 = False
        
        self.model = BGEM3Model(
            model_name=model_name_or_path, normlized=normalize_embeddings, sentence_pooling_method=pooling_method)
        if use_fp16:
            self.model.half()
        self.model = self.model.to(self.device)
        self.model.eval()

    def convert_id_to_token(self, lexical_weights: List[Dict]):
        if isinstance(lexical_weights, dict):
            lexical_weights = [lexical_weights]
        new_lexical_weights = []
        for item in lexical_weights:
            new_item = {}
            for id, weight in item.items():
                token = self.model.tokenizer.decode([int(id)])
                new_item[token] = weight
            new_lexical_weights.append(new_item)

        if len(new_lexical_weights) == 1:
            new_lexical_weights = new_lexical_weights[0]
        return new_lexical_weights

    def compute_lexical_matching_score(self, lexical_weights_1: Dict, lexical_weights_2: Dict):
        scores = 0
        for token, weight in lexical_weights_1.items():
            if token in lexical_weights_2:
                scores += weight * lexical_weights_2[token]
        return scores

    def colbert_score(self, q_reps, p_reps):
        q_reps, p_reps = torch.from_numpy(q_reps), torch.from_numpy(p_reps)
        token_scores = torch.einsum('in,jn->ij', q_reps, p_reps)
        scores, _ = token_scores.max(-1)
        scores = torch.sum(scores) / q_reps.size(0)
        return scores

    def __process_token_weights(self, token_weights: np.ndarray, input_ids: list):
            # conver to dict
            result = defaultdict(int)
            unused_tokens = set([self.tokenizer.cls_token_id, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                                 self.tokenizer.unk_token_id])
            # token_weights = np.ceil(token_weights * 100)
            for w, idx in zip(token_weights, input_ids):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    # w = int(w)
                    if w > result[idx]:
                        result[idx] = w
            return result

    def __process_colbert_vecs(self, colbert_vecs: np.ndarray, attention_mask: list):
        # delte the vectors of padding tokens
        tokens_num = np.sum(attention_mask)
        # we don't use the embedding of cls, so select tokens_num-1
        return colbert_vecs[:tokens_num - 1]

    
    def encode(self,
               sentences: Union[List[str], str],
               batch_size: int = 12,
               max_length: int = 8192,
               return_dense: bool = True,
               return_sparse: bool = False,
               return_colbert_vecs: bool = False) -> Dict:

        if self.num_gpus > 1:
            batch_size *= self.num_gpus
        self.model.eval()

        input_was_string = False
        if isinstance(sentences, str):
            sentences = [sentences]
            input_was_string = True

        all_dense_embeddings, all_lexical_weights, all_colbert_vec = [], [], []
        for start_index in tqdm(range(0, len(sentences), batch_size), desc="Inference Embeddings",
                                disable=len(sentences) < 256):
            sentences_batch = sentences[start_index:start_index + batch_size]
            batch_data = self.tokenizer(
                sentences_batch,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=max_length,
            ).to(self.device)
            output = self.model(batch_data,
                                return_dense=return_dense,
                                return_sparse=return_sparse,
                                return_colbert=return_colbert_vecs)
            if return_dense:
                all_dense_embeddings.append(output['dense_vecs'].cpu().numpy())

            if return_sparse:
                token_weights = output['sparse_vecs'].squeeze(-1)
                all_lexical_weights.extend(list(map(self.__process_token_weights, token_weights.cpu().numpy(),
                                                    batch_data['input_ids'].cpu().numpy().tolist())))

            if return_colbert_vecs:
                all_colbert_vec.extend(list(map(self.__process_colbert_vecs, output['colbert_vecs'].cpu().numpy(),
                                                batch_data['attention_mask'].cpu().numpy())))

        if return_dense:
            all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)

        if return_dense:
            if input_was_string:
                all_dense_embeddings = all_dense_embeddings[0]
        else:
            all_dense_embeddings = None

        if return_sparse:
            if input_was_string:
                all_lexical_weights = all_lexical_weights[0]
        else:
            all_lexical_weights = None

        if return_colbert_vecs:
            if input_was_string:
                all_colbert_vec = all_colbert_vec[0]
        else:
            all_colbert_vec = None

        return {"dense_vecs": all_dense_embeddings, "lexical_weights": all_lexical_weights,
                "colbert_vecs": all_colbert_vec}

    
    def tokenize(self, texts: Union[List[str], str], max_length: int):
        return self.model.tokenize(texts=texts, max_length=max_length)

    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 256,
                      max_query_length: int = 512,
                      max_passage_length: int = 8192) -> Dict[str, List[float]]:

        if self.num_gpus > 0:
            batch_size *= self.num_gpus
        self.model.eval()
        if isinstance(sentence_pairs, list) and len(sentence_pairs) == 0:
            return []
        if isinstance(sentence_pairs[0], str):
            one_input_pair = True
            sentence_pairs = [sentence_pairs]
        else:
            one_input_pair = False

        all_scores = {
            'colbert': [],
            'sparse': [],
            'dense': [],
            'sparse+dense': [],
            'colbert+sparse+dense': []
        }
        for start_index in tqdm(range(0, len(sentence_pairs), batch_size), desc="Compute Scores",
                                disable=len(sentence_pairs) < 128):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]

            queries_batch = [pair[0] for pair in sentences_batch]
            corpus_batch = [pair[1] for pair in sentences_batch]

            queries_inputs = self.tokenize(
                queries_batch, max_length=max_query_length).to(self.device)
            corpus_inputs = self.tokenize(
                corpus_batch, max_length=max_passage_length).to(self.device)

            queries_output = self.model(queries_inputs, return_dense=True, return_sparse=True, return_colbert=True,
                                        return_sparse_embedding=True)
            corpus_output = self.model(corpus_inputs, return_dense=True, return_sparse=True, return_colbert=True,
                                       return_sparse_embedding=True)

            q_dense_vecs, q_sparse_vecs, q_colbert_vecs = queries_output['dense_vecs'], queries_output['sparse_vecs'], \
                queries_output['colbert_vecs']
            p_dense_vecs, p_sparse_vecs, p_colbert_vecs = corpus_output['dense_vecs'], corpus_output['sparse_vecs'], \
                corpus_output['colbert_vecs']

            dense_scores = self.model.dense_score(q_dense_vecs, p_dense_vecs)
            sparse_scores = self.model.sparse_score(
                q_sparse_vecs, p_sparse_vecs)
            colbert_scores = self.model.colbert_score(q_colbert_vecs, p_colbert_vecs,
                                                      q_mask=queries_inputs['attention_mask'])
            
            inx = torch.arange(0, len(sentences_batch))
            dense_scores, sparse_scores, colbert_scores = dense_scores[inx, inx].float(), sparse_scores[
                inx, inx].float(), colbert_scores[inx, inx].float()

            all_scores['colbert'].extend(
                colbert_scores.cpu().numpy().tolist()
            )
            all_scores['sparse'].extend(
                sparse_scores.cpu().numpy().tolist()
            )
            all_scores['dense'].extend(
                dense_scores.cpu().numpy().tolist()
            )
            all_scores['sparse+dense'].extend(
                ((sparse_scores + dense_scores)/2).cpu().numpy().tolist()
            )
            all_scores['colbert+sparse+dense'].extend(
                ((colbert_scores + sparse_scores + dense_scores)/3).cpu().numpy().tolist()
            )

        if one_input_pair:
            return {k: v[0] for k, v in all_scores.items()}
        return all_scores
