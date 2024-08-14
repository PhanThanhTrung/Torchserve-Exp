rm -rf ./weights/flag_llm_reranker.mar
torch-model-archiver --model-name flag_llm_reranker --version 1.0 --model-file ./models/reranker/flag_llm_reranker.py --serialized-file ./weights/bge-reranker-v2-gemma/model-00001-of-00003.safetensors --export-path ./weights --handler ./models/reranker/flag_reranker_handler.py --extra-files "weights/bge-reranker-v2-gemma/config.json,weights/bge-reranker-v2-gemma/generation_config.json,weights/bge-reranker-v2-gemma/model-00002-of-00003.safetensors,weights/bge-reranker-v2-gemma/model-00003-of-00003.safetensors,weights/bge-reranker-v2-gemma/model.safetensors.index.json,weights/bge-reranker-v2-gemma/special_tokens_map.json,weights/bge-reranker-v2-gemma/tokenizer_config.json,weights/bge-reranker-v2-gemma/tokenizer.json,weights/bge-reranker-v2-gemma/tokenizer.model" --requirements-file "requirements.txt"
#  --archive-format "no-archive"