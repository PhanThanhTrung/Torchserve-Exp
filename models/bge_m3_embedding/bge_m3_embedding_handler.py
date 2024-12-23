import importlib
import logging
import os
from typing import List, Dict

import packaging
import torch
import transformers
from ts.torch_handler.base_handler import BaseHandler
from ts.utils.util import check_valid_pt2_backend, list_classes_from_module

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


class BGEM3EmbeddingHandler(BaseHandler):
    def __init__(self):
        super(BGEM3EmbeddingHandler, self).__init__()

    def initialize(self, context):
        if context is not None and hasattr(context, "model_yaml_config"):
            logger.info("Model YAML config not found.")
            self.model_yaml_config = context.model_yaml_config

        if self.model_yaml_config:
            self.setup_config = self.model_yaml_config.get("handler", {})
        else:
            logger.warning("Missing the handler config. Using a defalt config")
            self.setup_config = {
                "max_sequence_length": 2048
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

        model_file = self.manifest["model"].get("modelFile", "")

        if model_file:
            logger.debug("Loading eager model")
            self.model = self.load_model(
                model_dir=model_dir, model_file=model_file, model_pt_path=self.model_pt_path)
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

    def load_model(self, model_dir, model_file, model_pt_path):
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = list_classes_from_module(module)

        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        if model_pt_path:
            model = model_class(model_dir=model_dir)
            return model

    def preprocess(self, data: List[str]):
        """Basic text preprocessing.
        Args:
            data (str): The input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of prompts.
        """
        logger.info(data)
        if (isinstance(data, list) and len(data) == 0) or data is None:
            raise ValueError("Input should not be empty.")

        if isinstance(data, str):
            data = [data]
        else:
            data = data

        sentences_batch = []
        for input_text in data:
            logger.info("Received text: '%s'", input_text)
            if isinstance(input_text, dict):
                input_text = input_text.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")
            sentences_batch.append(input_text)
        
        max_sequence_length = int(
                self.setup_config.get("max_sequence_length"))
        sentences_batch: Dict = self.model.tokenize(
                sentences_batch, max_length=max_sequence_length)
        logger.info(
            f"Preprocessing is done.")
        logger.info(
            f"Output: {sentences_batch}.")
        return sentences_batch

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
