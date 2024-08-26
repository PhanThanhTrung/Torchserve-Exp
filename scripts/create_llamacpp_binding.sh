rm -rf ./weights/llama_cpp_binding.mar
torch-model-archiver --model-name llama_cpp_binding --version 1.0 --serialized-file ./weights/llama-8b-v3.1.gguf --export-path ./weights --handler models/llama_31/llama_31_handler.py  --requirements-file "requirements.txt"
#  --archive-format "no-archive"