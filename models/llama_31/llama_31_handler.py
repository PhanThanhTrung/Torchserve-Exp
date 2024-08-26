import json
import logging
import os
from typing import Dict, List

import llama_cpp
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Llama_cpp version %s", llama_cpp.__version__)

class LlamaCppHandler(BaseHandler):
    def __init__(self):
        super(LlamaCppHandler, self).__init__()
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
            _initial_prompt = "Please give me a short and correct answer. If your answer contain any command or statement in any language, please give me in the form of markdown format."
            self.setup_config = {
                "max_sequence_length": 8000,
                "initial_prompt": _initial_prompt,
                "use_fp16": True,
            }

        properties = context.system_properties
        self.manifest = context.manifest
        model_dir = properties.get("model_dir")
        self.model_pt_path = None
        if "serializedFile" in self.manifest["model"]:
            serialized_file = self.manifest["model"]["serializedFile"]
            self.model_pt_path = os.path.join(model_dir, serialized_file)

        if model_dir:
            logger.debug("Loading eager model")
            self.llm = llama_cpp.Llama(model_path=self.model_pt_path)
            self.sep_token = "\n"

        logger.debug("Model file %s loaded successfully", self.model_pt_path)
        self.initialized = True

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
            prompt = self.__check_and_convert(input_value=prompt, strict=False)
            query_input = self.__check_and_convert(input_value=query_input)

            prompt = prompt or initial_prompt
            _query_input = f"Q: {query_input}"
            input_sentence = prompt +  _query_input + "A: "
            batch_sentence.append(input_sentence)

        logger.info(msg=f"Done processing! Got: {batch_sentence}")
        return batch_sentence
    
    def inference(self, data: List[str], *args, **kwargs):
        """
        The Inference Function is used to make a prediction call on the given input request.
        The user needs to override the inference function to customize it.
        Args:
            data (Torch Tensor): A Torch Tensor is passed to make the Inference Request.
            The shape should match the model input shape.
        Returns:
            Torch Tensor : The Predicted Torch Tensor is returned in this function.
        """
        if isinstance(data, str):
            data = [data]
        outputs = []
        for prompt in data:
            _output = self.llm(prompt=prompt, 
                               max_tokens=None, 
                               echo=True, stop=["Q: "])
            outputs.append(_output)
        return outputs
    
    def postprocess(self, data: List[Dict]):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.
        Args:
            data: The torch tensor received from the prediction output of the model.
        Returns:
            List: The post process function returns a list of the predicted output.
        """
        return data