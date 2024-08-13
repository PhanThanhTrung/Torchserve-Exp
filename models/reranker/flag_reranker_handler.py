import logging
import os
from typing import Dict, List, Tuple

import packaging
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import check_valid_pt2_backend

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

if packaging.version.parse(torch.__version__) >= packaging.version.parse("2.0.0a"):
    PT2_AVAILABLE = True
    if torch.cuda.is_available():
        # If Ampere enable tensor cores which will give better performance
        # Ideally get yourself an A10G or A100 for optimal performance
        if torch.cuda.get_device_capability() >= (8, 0):
            torch.set_float32_matmul_precision("high")
            logger.info("Enabled tensor cores")
else:
    logger.warning(
        f"Your torch version is {torch.__version__} which does not support torch.compile"
    )
    PT2_AVAILABLE = False


class FlagLLMRerankerHandler(BaseHandler):
    """
    Flag LLM Reranker handler class for reranking sentence pairs.
    """
    def __init__(self):
        super(FlagLLMRerankerHandler, self).__init__()
        self.setup_config = None
        self.initialized = False
    
    def initialize(self, context):
        """In this initialize function, the FlagLLMReranker model is loaded.
        Args:
            context: It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        if context is not None and hasattr(context, "model_yaml_config"):
            logger.info("Model YAML config not found.")
            self.model_yaml_config = context.model_yaml_config

        if self.model_yaml_config:
            self.setup_config = self.model_yaml_config.get("handler", {})
        else:
            logger.warning("Missing the handler config. Using a defalt config")
            _default_prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            self.setup_config = {
                "max_sequence_length": 512,
                "initial_prompt": _default_prompt,
                "normalize": True,
            }

        properties = context.system_properties
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            logger.info(f"Detected CUDA. Setting model device is changed to gpu device id: {properties.get('gpu_id')}.")
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id")))
        elif torch.backends.mps.is_available() and properties.get("gpu_id") is not None:
            logger.info(f"Detected MPS backend. Setting model device is changed to mps device id: {properties.get('gpu_id')}.")
            self.map_location = "mps"
            self.device = torch.device("mps")
        else:
            logger.info(f"No GPU or MPS backend found. Setting model device is changed to cpu.")
            self.map_location = "cpu"
            self.device = torch.device(self.map_location)

        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)

        if model_dir:
            logger.debug("Loading eager model")
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForCausalLM.from_pretrained(model_dir)
            self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
            self.model.to(self.device)
            self.model.eval()

        if hasattr(self, "model_yaml_config") and "pt2" in self.model_yaml_config:
            pt2_value = self.model_yaml_config["pt2"]

            if "compile" in pt2_value:
                compile_options = pt2_value["compile"]
                if compile_options["enable"] == True:
                    del compile_options["enable"]

                    # if backend is not provided, compile will use its default, which is valid
                    valid_backend = (
                        check_valid_pt2_backend(compile_options["backend"])
                        if "backend" in compile_options
                        else True)
                else:
                    valid_backend = False
            elif "export" in pt2_value:
                valid_backend = False
            else:
                # pt2_value can be the backend, passed as a str, or arbitrary kwargs, passed as a dict
                if isinstance(pt2_value, str):
                    compile_options = dict(backend=pt2_value)
                elif isinstance(pt2_value, dict):
                    compile_options = pt2_value
                else:
                    raise ValueError("pt2 should be str or dict")

                # if backend is not provided, compile will use its default, which is valid
                valid_backend = (
                    check_valid_pt2_backend(compile_options["backend"])
                    if "backend" in compile_options
                    else True
                )

                logger.warning(
                    "This approach of specifying torch.compile() options is deprecated. The new standard approach is mentioned in https://github.com/pytorch/serve/issues/3164"
                )
        else:
            valid_backend = False

        if PT2_AVAILABLE and valid_backend:
            compile_options_str = ", ".join(
                [f"{k} {v}" for k, v in compile_options.items()]
            )
            # Compilation will delay your model initialization
            try:
                self.model = torch.compile(
                    self.model,
                    **compile_options,
                )
                logger.info(f"Compiled model with {compile_options_str}")
            except Exception as e:
                logger.warning(
                    f"Compiling model model with {compile_options_str} has failed \n Proceeding without compilation"
                )
                logger.warning(e)

        logger.debug("Model file %s loaded successfully", self.model_pt_path)
        self.initialized = True

    def preprocess(self, data: Dict):
        request_body = data.get("body")
        if request_body is None or not isinstance(request_body, dict):
            raise ValueError("Invalid Request!")
        
        prompt = request_body.get("prompt") or self.setup_config.get("initial_prompt")
        passage_input = request_body.get("passage")
        query_input = request_body.get("query")
        
        prompt = self.__check_and_convert(input_bytes=prompt)
        passage_input = self.__check_and_convert(input_bytes=passage_input)
        query_input = self.__check_and_convert(input_bytes=query_input)
        
        sentences_pair = [passage_input, query_input]

        prompt_inputs = self.tokenizer(prompt,
                                       return_tensors=None,
                                       add_special_tokens=False)["input_ids"]
        
        encode_max_length = self.setup_config.get("max_sequence_length") + len(sep_inputs) + len(prompt_inputs)
        
        sep = "\n"
        sep_inputs = self.tokenizer(sep,
                                    return_tensors=None,
                                    add_special_tokens=False)["input_ids"]
        query_input = [f"A: {query_input}"]
        passage_input = [f"B: {passage_input}"]

        queries_inputs = self.tokenizer(query_input,
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=encode_max_length * 3 // 4,
                                        truncation=True)["input_ids"]
        passages_inputs = self.tokenizer(passage_input,
                                        return_tensors=None,
                                        add_special_tokens=False,
                                        max_length=encode_max_length,
                                        truncation=True)["input_ids"]
        batch_input = []
        for _query_input, _passage_input in zip(queries_inputs, passages_inputs):
            item = self.tokenizer.prepare_for_model(
                        [self.tokenizer.bos_token_id] + _query_input,
                        sep_inputs + _passage_input,
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
            batch_input.append(item)

        logger.info(f"Preprocessing is done.")
        logger.info(f"Output: {sentences_pair}.")
        return sentences_pair

    def __check_and_convert(self, input_bytes):
        if isinstance(input_bytes, (bytearray, bytes)):
            input_bytes = input_bytes.decode("utf-8")
            return input_bytes
        else:
            raise ValueError("Invalid Request!")

    @torch.inference_mode
    def inference(self, data: Dict, *args, **kwargs):
        """Receive a Dictionary which contains output of tokenizer. Responsible for passing those output
        into embedding model.
        Args:
            data (Dict): Dictionary contains output of tokenizer.
        Returns: It returns a  of the embeddings for the input text
        """
        logger.info(f"Process embedding.")
        _output = self.model(data, return_dense=True, return_sparse=True,
                                return_colbert=True, return_sparse_embedding=True)
        return _output

    def postprocess(self, data):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        if isinstance(data, dict):
            for key in data:
                data[key] = data[key].cpu().numpy().tolist()
            data = [data]
        return data