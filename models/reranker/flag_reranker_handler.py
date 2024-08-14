import json
import logging
import os
from typing import Dict

import numpy as np
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
        f"Your torch version is {
            torch.__version__} which does not support torch.compile"
    )
    PT2_AVAILABLE = False


class FlagLLMRerankerHandler(BaseHandler):
    def __init__(self):
        super(FlagLLMRerankerHandler, self).__init__()
        self.setup_config = None
        self.initialized = False

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           First try to load torchscript else load eager mode state_dict based model.
        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """

        if context is not None and hasattr(context, "model_yaml_config"):
            logger.info("Model YAML config not found.")
            self.model_yaml_config = context.model_yaml_config

        if self.model_yaml_config:
            self.setup_config = self.model_yaml_config.get("handler", {})
        else:
            logger.warning("Missing the handler config. Using a defalt config")
            _initial_prompt = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
            self.setup_config = {
                "max_sequence_length": 8000,
                "initial_prompt": _initial_prompt,
                "use_fp16": True,
            }

        properties = context.system_properties
        if torch.cuda.is_available() and properties.get("gpu_id") is not None:
            logger.info(f"Detected CUDA. Setting model device is changed to gpu device id: {
                        properties.get('gpu_id')}.")
            self.map_location = "cuda"
            self.device = torch.device(
                self.map_location + ":" + str(properties.get("gpu_id")))
        elif torch.backends.mps.is_available() and properties.get("gpu_id") is not None:
            logger.info(f"Detected MPS backend. Setting model device is changed to mps device id: {
                        properties.get('gpu_id')}.")
            self.map_location = "mps"
            self.device = torch.device("mps")
        else:
            logger.info(
                f"No GPU or MPS backend found. Setting model device is changed to cpu.")
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
            if self.setup_config.get("use_fp16"):
                self.model.half()
            self.model.to(self.device)
            self.model.eval()

            self.yes_loc = self.tokenizer("Yes", add_special_tokens=False)[
                "input_ids"][0]
            self.sep_token = "\n"

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
                    f"Compiling model model with {
                        compile_options_str} has failed \n Proceeding without compilation"
                )
                logger.warning(e)

        logger.debug("Model file %s loaded successfully", self.model_pt_path)
        self.initialized = True

    def preprocess(self, data: Dict) -> Dict:
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing
        Args :
            data (list): List of the data from the request input.
        Returns:
            tensor: Returns the tensor data of the input
        """
        logger.info(msg=f"Current request data: {data}")
        if isinstance(data, dict):
            if data.get("body") is None:
                raise ValueError("Request is invalid!")
            data = [data]

        if isinstance(data, list) and len(data) == 0:
            raise ValueError("Request is invalid!")

        max_length = self.setup_config.get("max_sequence_length")
        initial_prompt = self.setup_config.get("initial_prompt")
        batch_sentence = []
        for _data in data:
            request_body = _data.get("body")
            if isinstance(request_body, (bytes, bytearray)):
                request_body = request_body.decode("utf-8")

            if isinstance(request_body, str):
                request_body = json.loads(request_body)

            prompt = request_body.get("prompt")
            query_input = request_body.get("query")
            passage_input = request_body.get("passage")
            prompt = self.__check_and_convert(input_value=prompt, strict=False)
            query_input = self.__check_and_convert(input_value=query_input)
            passage_input = self.__check_and_convert(input_value=passage_input)

            prompt = prompt or initial_prompt
            _query_input = f"A: {query_input}"
            _passage_input = f"B: {passage_input}"
            input_sentence = _query_input + self.sep_token + \
                _passage_input + self.sep_token + prompt
            batch_sentence.append(input_sentence)

        logger.info(msg="Starting tokenize input batch!")
        tokenized_input_sentence = self.tokenizer(text=input_sentence, add_special_tokens=True,
                                                  truncation=True, padding=True, max_length=max_length, return_attention_mask=True, return_tensors="pt")
        tokenized_input_sentence = tokenized_input_sentence.to(self.device)
        return tokenized_input_sentence

    def __check_and_convert(self, input_value, strict=False):
        if isinstance(input_value, (bytes, bytearray)):
            input_value = input_value.decode("utf-8")
            return input_value
        elif isinstance(input_value, str):
            return input_value
        else:
            if strict:
                raise ValueError("Request is invalid!")
            else:
                return None

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

    @torch.inference_mode
    def inference(self, data: Dict, *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        _cur_attention_mask = data.get("attention_mask")
        outputs = self.model(**data, output_hidden_states=True)
        logits = outputs.logits
        raw_scores = self.__last_logit_pool(logits, _cur_attention_mask)
        raw_score = raw_scores[:, self.yes_loc]
        normalized_score = self.__sigmoid(x=raw_score)

        return normalized_score

    def postprocess(self, data: torch.Tensor):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data (Torch Tensor): The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """
        return data.cpu().tolist()
