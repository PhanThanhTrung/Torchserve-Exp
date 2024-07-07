import logging
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from FlagEmbedding.BGE_M3 import BGEM3ForInference
from huggingface_hub import snapshot_download
from torch import Tensor, nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, is_torch_npu_available

logger = logging.getLogger(__name__)


class BGEM3Model(nn.Module):
    def __init__(self,
                 model_name: str = None,
                 normlized: bool = True,
                 sentence_pooling_method: str = 'cls',
                 unified_finetuning: bool = True,
                 colbert_dim: int = -1):
        super().__init__()
        self.load_model(model_name, colbert_dim=colbert_dim)
        self.vocab_size: int = self.model.config.vocab_size

        self.unified_finetuning: bool = unified_finetuning
        if not self.unified_finetuning:
            self.colbert_linear: Optional[nn.Module] = None
            self.sparse_linear: Optional[nn.Module] = None

        self.normlized: bool = normlized
        self.sentence_pooling_method: str = sentence_pooling_method

    def load_model(self, model_name, colbert_dim: int = -1):
        if not os.path.exists(model_name):
            cache_folder = os.getenv('HF_HUB_CACHE')
            model_name = snapshot_download(repo_id=model_name,
                                           cache_dir=cache_folder,
                                           ignore_patterns=['flax_model.msgpack', 'rust_model.ot', 'tf_model.h5'])

        self.model: AutoModel = AutoModel.from_pretrained(model_name)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            model_name)

        self.colbert_linear: nn.Linear = torch.nn.Linear(in_features=self.model.config.hidden_size,
                                                         out_features=self.model.config.hidden_size if colbert_dim == -1 else colbert_dim)
        self.sparse_linear: nn.Linear = torch.nn.Linear(
            in_features=self.model.config.hidden_size, out_features=1)

        if os.path.exists(os.path.join(model_name, 'colbert_linear.pt')) and os.path.exists(
                os.path.join(model_name, 'sparse_linear.pt')):
            logger.info(
                'Loading existing colbert_linear and sparse_linear---------')
            self.load_pooler(model_dir=model_name)
        else:
            logger.info(
                'The parameters of colbert_linear and sparse linear is new initialize. Make sure the model is loaded for training, not inferencing')

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

    def _encode(self, features):
        dense_vecs, sparse_vecs, colbert_vecs = None, None, None
        last_hidden_state = self.model(
            **features, return_dict=True).last_hidden_state
        dense_vecs = self.dense_embedding(
            last_hidden_state, features['attention_mask'])
        if self.unified_finetuning:
            sparse_vecs = self.sparse_embedding(
                last_hidden_state, features['input_ids'])
            colbert_vecs = self.colbert_embedding(
                last_hidden_state, features['attention_mask'])
        if self.normlized:
            dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)
            if self.unified_finetuning:
                colbert_vecs = torch.nn.functional.normalize(
                    colbert_vecs, dim=-1)
        return dense_vecs, sparse_vecs, colbert_vecs

    def encode(self, features, sub_batch_size=None):
        if features is None:
            return None

        if sub_batch_size is not None and sub_batch_size != -1:
            all_dense_vecs, all_sparse_vecs, all_colbert_vecs = [], [], []
            for i in range(0, len(features['attention_mask']), sub_batch_size):
                end_inx = min(i + sub_batch_size,
                              len(features['attention_mask']))
                sub_features = {}
                for k, v in features.items():
                    sub_features[k] = v[i:end_inx]

                dense_vecs, sparse_vecs, colbert_vecs = self._encode(
                    sub_features)
                all_dense_vecs.append(dense_vecs)
                all_sparse_vecs.append(sparse_vecs)
                all_colbert_vecs.append(colbert_vecs)

            dense_vecs = torch.cat(all_dense_vecs, 0)
            if self.unified_finetuning:
                sparse_vecs = torch.cat(all_sparse_vecs, 0)
                colbert_vecs = torch.cat(all_colbert_vecs, 0)
        else:
            dense_vecs, sparse_vecs, colbert_vecs = self._encode(features)

        if self.unified_finetuning:
            return dense_vecs.contiguous(), sparse_vecs.contiguous(), colbert_vecs.contiguous()
        else:
            return dense_vecs.contiguous(), None, None

    @classmethod
    def compute_similarity(cls, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def load_pooler(self, model_dir):
        colbert_state_dict = torch.load(os.path.join(
            model_dir, 'colbert_linear.pt'), map_location='cpu')
        sparse_state_dict = torch.load(os.path.join(
            model_dir, 'sparse_linear.pt'), map_location='cpu')
        self.colbert_linear.load_state_dict(colbert_state_dict)
        self.sparse_linear.load_state_dict(sparse_state_dict)


class BGEM3FlagModel:
    def __init__(
            self,
            model_name_or_path: str = None,
            pooling_method: str = 'cls',
            normalize_embeddings: bool = True,
            use_fp16: bool = True,
            device: str = None
    ) -> None:

        self.model = BGEM3ForInference(
            model_name=model_name_or_path,
            normlized=normalize_embeddings,
            sentence_pooling_method=pooling_method,
        )

        self.tokenizer = self.model.tokenizer
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
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

        if device is None:
            self.num_gpus = torch.cuda.device_count()
            if self.num_gpus > 1:
                print(f"----------using {self.num_gpus}*GPUs----------")
                self.model.model = torch.nn.DataParallel(self.model.model)
        else:
            self.num_gpus = 1

        self.model.eval()

    def convert_id_to_token(self, lexical_weights: List[Dict]):
        if isinstance(lexical_weights, dict):
            lexical_weights = [lexical_weights]
        new_lexical_weights = []
        for item in lexical_weights:
            new_item = {}
            for id, weight in item.items():
                token = self.tokenizer.decode([int(id)])
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

    @torch.no_grad()
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

        def _process_token_weights(token_weights: np.ndarray, input_ids: list):
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

        def _process_colbert_vecs(colbert_vecs: np.ndarray, attention_mask: list):
            # delte the vectors of padding tokens
            tokens_num = np.sum(attention_mask)
            # we don't use the embedding of cls, so select tokens_num-1
            return colbert_vecs[:tokens_num - 1]

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
                all_lexical_weights.extend(list(map(_process_token_weights, token_weights.cpu().numpy(),
                                                    batch_data['input_ids'].cpu().numpy().tolist())))

            if return_colbert_vecs:
                all_colbert_vec.extend(list(map(_process_colbert_vecs, output['colbert_vecs'].cpu().numpy(),
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

    @torch.no_grad()
    def compute_score(self,
                      sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
                      batch_size: int = 256,
                      max_query_length: int = 512,
                      max_passage_length: int = 8192,
                      weights_for_different_modes: List[float] = None) -> Dict[str, List[float]]:

        def _tokenize(texts: list, max_length: int):
            return self.tokenizer(
                texts,
                max_length=max_length,
                padding=True,
                return_token_type_ids=False,
                truncation=True,
                return_tensors='pt'
            )

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

            queries_inputs = _tokenize(
                queries_batch, max_length=max_query_length).to(self.device)
            corpus_inputs = _tokenize(
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

            if weights_for_different_modes is None:
                weights_for_different_modes = [1, 1., 1.]
                weight_sum = 3
                print(
                    "default weights for dense, sparse, colbert are [1.0, 1.0, 1.0] ")
            else:
                assert len(weights_for_different_modes) == 3
                weight_sum = sum(weights_for_different_modes)

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
                ((sparse_scores * weights_for_different_modes[1] + dense_scores * weights_for_different_modes[0])/(
                    weights_for_different_modes[1]+weights_for_different_modes[0])).cpu().numpy().tolist()
            )
            all_scores['colbert+sparse+dense'].extend(
                ((colbert_scores * weights_for_different_modes[2] + sparse_scores * weights_for_different_modes[1] +
                 dense_scores * weights_for_different_modes[0])/weight_sum).cpu().numpy().tolist()
            )

        if one_input_pair:
            return {k: v[0] for k, v in all_scores.items()}
        return all_scores
