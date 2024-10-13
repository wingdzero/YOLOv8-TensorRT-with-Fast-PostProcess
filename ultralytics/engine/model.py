# Ultralytics YOLO 🚀, AGPL-3.0 license

import inspect
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.engine.results import Results
from ultralytics.hub import HUB_WEB_ROOT, HUBTrainingSession
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, nn, yaml_model_load
from ultralytics.utils import (
    ARGV,
    ASSETS,
    DEFAULT_CFG_DICT,
    LOGGER,
    RANK,
    callbacks,
    checks,
    emojis,
    yaml_load,
)


class Model(nn.Module):
    """
        一个用于实现 YOLO 模型的基类，统一不同模型类型的 API
        该类为 YOLO 模型相关的各种操作（如训练、验证、预测、导出和基准测试）提供了一个通用接口。它处理不同类型的模型，
        包括从本地文件、Ultralytics HUB 或 Triton Server 加载的模型。该类设计灵活且可扩展，适用于不同的任务和模型配置

        参数：
            model（Union[str, Path], 可选）：要加载或创建的模型路径或名称。可以是本地文件路径、来自 Ultralytics HUB 的模型名称或 Triton Server 模型。默认为 'yolov8n.pt'
            task（Any, 可选）：与 YOLO 模型关联的任务类型。可以用于指定模型的应用领域，如目标检测、分割等。默认为 None
            verbose（bool, 可选）：如果为 True，则在模型操作期间启用详细输出。默认为 False

        属性：
            callbacks（dict）：在模型操作期间用于各种事件的回调函数字典
            predictor（BasePredictor）：用于进行预测的预测器对象
            model（nn.Module）：基础的 PyTorch 模型
            trainer（BaseTrainer）：用于训练模型的训练器对象
            ckpt（dict）：如果从 *.pt 文件加载模型，则为检查点数据
            cfg（str）：如果从 *.yaml 文件加载模型，则为模型配置
            ckpt_path（str）：检查点文件的路径
            overrides（dict）：模型配置的覆盖字典
            metrics（dict）：最新的训练/验证指标
            session（HUBTrainingSession）：如果适用，为 Ultralytics HUB 会话
            task（str）：模型的任务类型
            model_name（str）：模型的名称

        方法：
            call：预测方法的别名，使模型实例可调用
            _new：基于配置文件初始化新模型
            _load：从检查点文件加载模型
            _check_is_pytorch_model：确保模型是 PyTorch 模型
            reset_weights：将模型的权重重置为初始状态
            load：从指定文件加载模型权重
            save：将模型的当前状态保存到文件
            info：记录或返回有关模型的信息
            fuse：融合 Conv2d 和 BatchNorm2d 层以优化推理
            predict：执行目标检测预测
            track：执行目标跟踪
            val：在数据集上验证模型
            benchmark：对各种导出格式进行基准测试
            export：将模型导出为不同格式
            train：在数据集上训练模型
            tune：执行超参数调优
            _apply：对模型的张量应用函数
            add_callback：为事件添加回调函数
            clear_callback：清除事件的所有回调
            reset_callbacks：将所有回调重置为默认函数
            is_triton_model：检查模型是否为 Triton Server 模型
            is_hub_model：检查模型是否为 Ultralytics HUB 模型
            _reset_ckpt_args：加载 PyTorch 模型时重置检查点参数
            _smart_load：根据模型任务加载适当的模块
            task_map：提供从模型任务到相应类的映射

        异常：
            FileNotFoundError：如果指定的模型文件不存在或不可访问
            ValueError：如果模型文件或配置无效或不受支持
            ImportError：如果未安装特定模型类型所需的依赖项（如 HUB SDK）
            TypeError：如果模型不是所需的 PyTorch 模型
            AttributeError：如果所需的属性或方法未实现或不可用
            NotImplementedError：如果不支持特定的模型任务或模式
    """

    def __init__(
        self,
        model: Union[str, Path] = "yolov8n.pt",
        task: str = None,
        verbose: bool = False,
    ) -> None:
        """
        初始化一个新的Model类实例。

        这个构造函数根据提供的模型路径或名称来设置模型。它处理各种类型的模型来源，包括本地文件、Ultralytics HUB模型和Triton Server模型。该方法初始化模型的几个重要属性，并为训练、预测或导出等操作做好准备。

        参数：

        model (Union[str, Path], optional): 要加载或创建的模型文件的路径或名称。这可以是本地文件路径、来自Ultralytics HUB的模型名称或Triton Server模型。默认为 'yolov8n.pt'。
        task (Any, optional): 与YOLO模型相关的任务类型，指定其应用领域。默认为None。
        verbose (bool, optional): 如果为True，则在模型初始化和后续操作期间启用详细输出。默认为False。
        异常：

        FileNotFoundError: 如果指定的模型文件不存在或无法访问。
        ValueError: 如果模型文件或配置无效或不受支持。
        ImportError: 如果特定模型类型（如HUB SDK）的必需依赖项未安装。
        """
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None  # reuse predictor
        self.model = None  # model object
        self.trainer = None  # trainer object
        self.ckpt = None  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml
        self.ckpt_path = None
        self.overrides = {}  # overrides for trainer object
        self.metrics = None  # validation/training metrics
        self.session = None  # HUB session
        self.task = task  # task type
        model = str(model).strip()  # .strip()方法用于自动移除str两端字符，默认是移除空格字符

        # Check if Ultralytics HUB model from https://hub.ultralytics.com
        if self.is_hub_model(model):
            # Fetch model from HUB
            checks.check_requirements("hub-sdk>=0.0.8")
            self.session = HUBTrainingSession.create_session(model)
            model = self.session.model_file

        # Check if Triton Server model
        elif self.is_triton_model(model):
            self.model_name = self.model = model
            return

        # Load or create new YOLO model
        if Path(model).suffix in {".yaml", ".yml"}:
            # 基于yaml新建模型
            self._new(model, task=task, verbose=verbose)
        else:
            # 加载pt模型
            self._load(model, task=task)

    def __call__(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        """
        An alias for the predict method, enabling the model instance to be callable.

        This method simplifies the process of making predictions by allowing the model instance to be called directly
        with the required arguments for prediction.

        Args:
            source (str | Path | int | PIL.Image | np.ndarray, optional): The source of the image for making
                predictions. Accepts various types, including file paths, URLs, PIL images, and numpy arrays.
                Defaults to None.
            stream (bool, optional): If True, treats the input source as a continuous stream for predictions.
                Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the prediction process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of prediction results, encapsulated in the Results class.
        """
        return self.predict(source, stream, **kwargs)

    @staticmethod
    def is_triton_model(model: str) -> bool:
        """Is model a Triton Server URL string, i.e. <scheme>://<netloc>/<endpoint>/<task_name>"""
        from urllib.parse import urlsplit

        url = urlsplit(model)
        return url.netloc and url.path and url.scheme in {"http", "grpc"}

    @staticmethod
    def is_hub_model(model: str) -> bool:
        """Check if the provided model is a HUB model."""
        return any(
            (
                model.startswith(f"{HUB_WEB_ROOT}/models/"),  # i.e. https://hub.ultralytics.com/models/MODEL_ID
                [len(x) for x in model.split("_")] == [42, 20],  # APIKEY_MODEL
                len(model) == 20 and not Path(model).exists() and all(x not in model for x in "./\\"),  # MODEL
            )
        )

    def _new(self, cfg: str, task=None, model=None, verbose=False) -> None:
        """
        Initializes a new model and infers the task type from the model definitions.

        Args:
            cfg (str): model configuration file
            task (str | None): model task
            model (BaseModel): Customized model.
            verbose (bool): display model info on load
        """
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load("model"))(cfg_dict, verbose=verbose and RANK == -1)  # build model
        self.overrides["model"] = self.cfg
        self.overrides["task"] = self.task

        # Below added to allow export from YAMLs
        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}  # combine default and model args (prefer model args)
        self.model.task = self.task
        self.model_name = cfg

    def _load(self, weights: str, task=None) -> None:
        """
        初始化一个新模型并从模型的head推断任务类型。

        参数：

        weights (str): 要加载的模型检查点。
        task (str | None): 模型任务。
        """
        if weights.lower().startswith(("https://", "http://", "rtsp://", "rtmp://", "tcp://")):
            weights = checks.check_file(weights)  # automatically download and return local filename
        weights = checks.check_model_file_from_stem(weights)  # add suffix, i.e. yolov8n -> yolov8n.pt

        if Path(weights).suffix == ".pt":
            self.model, self.ckpt = attempt_load_one_weight(weights)  # 基于torch.load()加载模型
            self.task = self.model.args["task"]
            self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)  # 提取"imgsz", "data", "task", "single_cls"，only remember these arguments when loading a PyTorch model
            self.ckpt_path = self.model.pt_path
        else:
            weights = checks.check_file(weights)  # runs in all cases, not redundant with above call
            self.model, self.ckpt = weights, None
            self.task = task or guess_model_task(weights)
            self.ckpt_path = weights
        self.overrides["model"] = weights
        self.overrides["task"] = self.task
        self.model_name = weights

    def _check_is_pytorch_model(self) -> None:
        """Raises TypeError is model is not a PyTorch model."""
        pt_str = isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pt"  # 检查后缀
        pt_module = isinstance(self.model, nn.Module)  # 检查model是否是nn.Module类对象
        if not (pt_module or pt_str):
            raise TypeError(
                f"model='{self.model}' should be a *.pt PyTorch model to run this method, but is a different format. "
                f"PyTorch models can train, val, predict and export, i.e. 'model.train(data=...)', but exported "
                f"formats like ONNX, TensorRT etc. only support 'predict' and 'val' modes, "
                f"i.e. 'yolo predict model=yolov8n.onnx'.\nTo run CUDA or MPS inference please pass the device "
                f"argument directly in your inference command, i.e. 'model.predict(source=..., device=0)'"
            )

    def reset_weights(self) -> "Model":
        """
        Resets the model parameters to randomly initialized values, effectively discarding all training information.

        This method iterates through all modules in the model and resets their parameters if they have a
        'reset_parameters' method. It also ensures that all parameters have 'requires_grad' set to True, enabling them
        to be updated during training.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with reset weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights: Union[str, Path] = "yolov8n.pt") -> "Model":
        """
        Loads parameters from the specified weights file into the model.

        This method supports loading weights from a file or directly from a weights object. It matches parameters by
        name and shape and transfers them to the model.

        Args:
            weights (str | Path): Path to the weights file or a weights object. Defaults to 'yolov8n.pt'.

        Returns:
            self (ultralytics.engine.model.Model): The instance of the class with loaded weights.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def save(self, filename: Union[str, Path] = "saved_model.pt", use_dill=True) -> None:
        """
        Saves the current model state to a file.

        This method exports the model's checkpoint (ckpt) to the specified filename.

        Args:
            filename (str | Path): The name of the file to save the model to. Defaults to 'saved_model.pt'.
            use_dill (bool): Whether to try using dill for serialization if available. Defaults to True.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from copy import deepcopy
        from datetime import datetime

        from ultralytics import __version__

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        }
        torch.save({**self.ckpt, **updates}, filename, use_dill=use_dill)

    def info(self, detailed: bool = False, verbose: bool = True):
        """
        Logs or returns model information.

        This method provides an overview or detailed information about the model, depending on the arguments passed.
        It can control the verbosity of the output.

        Args:
            detailed (bool): If True, shows detailed information about the model. Defaults to False.
            verbose (bool): If True, prints the information. If False, returns the information. Defaults to True.

        Returns:
            (list): Various types of information about the model, depending on the 'detailed' and 'verbose' parameters.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        """
        Fuses Conv2d and BatchNorm2d layers in the model.

        This method optimizes the model by fusing Conv2d and BatchNorm2d layers, which can improve inference speed.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        self.model.fuse()

    def embed(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        **kwargs,
    ) -> list:
        """
        Generates image embeddings based on the provided source.

        This method is a wrapper around the 'predict()' method, focusing on generating embeddings from an image source.
        It allows customization of the embedding process through various keyword arguments.

        Args:
            source (str | int | PIL.Image | np.ndarray): The source of the image for generating embeddings.
                The source can be a file path, URL, PIL image, numpy array, etc. Defaults to None.
            stream (bool): If True, predictions are streamed. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the embedding process.

        Returns:
            (List[torch.Tensor]): A list containing the image embeddings.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        if not kwargs.get("embed"):
            kwargs["embed"] = [len(self.model.model) - 2]  # embed second-to-last layer if no indices passed
        return self.predict(source, stream, **kwargs)

    def predict(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        predictor=None,
        **kwargs,
    ) -> List[Results]:
        """
        对给定的图像源使用 YOLO 模型进行预测。

        这个方法简化了预测过程，允许通过关键字参数进行各种配置。它支持使用自定义预测器或默认预测器方法进行预测。该方法处理不同类型的图像源，并且可以以流模式运行。它还通过'prompts'支持SAM类型模型。

        如果尚未存在预测器，该方法会设置一个新的预测器，并在每次调用时更新其参数。如果未提供'source'，该方法会发出警告并使用默认资源。该方法确定是否从命令行界面调用，并相应地调整其行为，包括设置置信度阈值和保存行为的默认值。

        参数：

        source (str | int | PIL.Image | np.ndarray, optional): 用于进行预测的图像源。接受各种类型，包括文件路径、URL、PIL图像和numpy数组。默认为ASSETS。
        stream (bool, optional): 将输入源视为连续流进行预测。默认为False。
        predictor (BasePredictor, optional): 用于进行预测的自定义预测器类的实例。如果为None，方法使用默认预测器。默认为None。
        **kwargs (any): 配置预测过程的其他关键字参数。这些参数允许进一步自定义预测行为。
        返回：

        (List[ultralytics.engine.results.Results]): 一个预测结果列表，封装在Results类中。
        引发：

        AttributeError: 如果预测器未正确设置。
        """
        if source is None:
            source = ASSETS
            LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using 'source={source}'.")

        is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
            x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
        )

        custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict"}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
        prompts = args.pop("prompts", None)  # for SAM-type models

        if not self.predictor:
            self.predictor = predictor or self._smart_load("predictor")(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=is_cli)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        if prompts and hasattr(self.predictor, "set_prompts"):  # for SAM-type models
            self.predictor.set_prompts(prompts)
        return self.predictor.predict_cli(source=source) if is_cli else self.predictor(source=source, stream=stream)

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It is
        capable of handling different types of input sources such as file paths or video streams. The method supports
        customization of the tracking process through various keyword arguments. It registers trackers if they are not
        already present and optionally persists them based on the 'persist' flag.

        The method sets a default confidence threshold specifically for ByteTrack-based tracking, which requires low
        confidence predictions as input. The tracking mode is explicitly set in the keyword arguments.

        Args:
            source (str, optional): The input source for object tracking. It can be a file path, URL, or video stream.
            stream (bool, optional): Treats the input source as a continuous video stream. Defaults to False.
            persist (bool, optional): Persists the trackers between different calls to this method. Defaults to False.
            **kwargs (any): Additional keyword arguments for configuring the tracking process. These arguments allow
                for further customization of the tracking behavior.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, encapsulated in the Results class.

        Raises:
            AttributeError: If the predictor does not have registered trackers.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker

            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)

    def val(
        self,
        validator=None,
        **kwargs,
    ):
        """
        Validates the model using a specified dataset and validation configuration.

        This method facilitates the model validation process, allowing for a range of customization through various
        settings and configurations. It supports validation with a custom validator or the default validation approach.
        The method combines default configurations, method-specific defaults, and user-provided arguments to configure
        the validation process. After validation, it updates the model's metrics with the results obtained from the
        validator.

        The method supports various arguments that allow customization of the validation process. For a comprehensive
        list of all configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            validator (BaseValidator, optional): An instance of a custom validator class for validating the model. If
                None, the method uses a default validator. Defaults to None.
            **kwargs (any): Arbitrary keyword arguments representing the validation configuration. These arguments are
                used to customize various aspects of the validation process.

        Returns:
            (ultralytics.utils.metrics.DetMetrics): Validation metrics obtained from the validation process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        custom = {"rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def benchmark(
        self,
        **kwargs,
    ):
        """
        Benchmarks the model across various export formats to evaluate performance.

        This method assesses the model's performance in different export formats, such as ONNX, TorchScript, etc.
        It uses the 'benchmark' function from the ultralytics.utils.benchmarks module. The benchmarking is configured
        using a combination of default configuration values, model-specific arguments, method-specific defaults, and
        any additional user-provided keyword arguments.

        The method supports various arguments that allow customization of the benchmarking process, such as dataset
        choice, image size, precision modes, device selection, and verbosity. For a comprehensive list of all
        configurable options, users should refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the benchmarking process. These are combined with
                default configurations, model-specific arguments, and method defaults.

        Returns:
            (dict): A dictionary containing the results of the benchmarking process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from ultralytics.utils.benchmarks import benchmark

        custom = {"verbose": False}  # method defaults
        args = {**DEFAULT_CFG_DICT, **self.model.args, **custom, **kwargs, "mode": "benchmark"}
        return benchmark(
            model=self,
            data=kwargs.get("data"),  # if no 'data' argument passed set data=None for default datasets
            imgsz=args["imgsz"],
            half=args["half"],
            int8=args["int8"],
            device=args["device"],
            verbose=kwargs.get("verbose"),
        )

    def export(
        self,
        **kwargs,
    ) -> str:
        """
        Exports the model to a different format suitable for deployment.

        This method facilitates the export of the model to various formats (e.g., ONNX, TorchScript) for deployment
        purposes. It uses the 'Exporter' class for the export process, combining model-specific overrides, method
        defaults, and any additional arguments provided. The combined arguments are used to configure export settings.

        The method supports a wide range of arguments to customize the export process. For a comprehensive list of all
        possible arguments, refer to the 'configuration' section in the documentation.

        Args:
            **kwargs (any): Arbitrary keyword arguments to customize the export process. These are combined with the
                model's overrides and method defaults.

        Returns:
            (str): The exported model filename in the specified format, or an object related to the export process.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        from .exporter import Exporter

        custom = {
            "imgsz": self.model.args["imgsz"],
            "batch": 1,
            "data": None,
            "device": None,  # reset to avoid multi-GPU errors
            "verbose": False,
        }  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "export"}  # highest priority args on the right
        return Exporter(overrides=args, _callbacks=self.callbacks)(model=self.model)

    def train(
        self,
        trainer=None,
        **kwargs,  # {'batch': 4, 'data': './pipe.yaml', 'epochs': 300, 'imgsz': 1500, 'max_det': 2000, 'workers': 1}
    ):
        """
        使用指定的数据集和训练配置训练模型。

        此方法通过一系列可定制的设置和配置来促进模型训练。它支持使用自定义的训练器或方法中定义的默认训练方法进行训练。
        该方法处理不同的情况，例如从检查点恢复训练、与 Ultralytics HUB 集成以及在训练后更新模型和配置。
        当使用 Ultralytics HUB 时，如果会话已经加载了模型，则方法优先考虑 HUB 的训练参数，并在提供本地参数时发出警告。
        它检查 pip 更新，并结合默认配置、方法特定的默认值和用户提供的参数来配置训练过程。训练后，它更新模型及其配置，并可选择性地附加度量指标。

        参数：
            trainer (BaseTrainer, optional): 自定义训练器类的实例，用于训练模型。如果为 None，则方法使用默认训练器。默认值为 None。
            **kwargs (any): 表示训练配置的任意关键字参数。这些参数用于定制训练过程的各个方面。
        返回值：
            (dict | None): 如果可用且训练成功，则返回训练指标；否则返回 None。
        异常：
            AssertionError: 如果模型不是 PyTorch 模型。
            PermissionError: 如果 HUB 会话存在权限问题。
            ModuleNotFoundError: 如果未安装 HUB SDK。
        """
        self._check_is_pytorch_model()
        if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
            if any(kwargs):
                LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
            kwargs = self.session.train_args  # overwrite kwargs

        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides  # {'data': 'coco.yaml', 'imgsz': 640, 'model': './yolov8m.pt', 'single_cls': False, 'task': 'detect'}
        custom = {
            # NOTE: handle the case when 'cfg' includes 'data'.
            "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
            "model": self.overrides["model"],
            "task": self.task,
        }  # method defaults
        args = {**overrides, **custom, **kwargs, "mode": "train"}  # 合并多个参数字典时，最右边的字典具有最高优先级。
        if args.get("resume"):
            args["resume"] = self.ckpt_path

        self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)  # 初始化训练器，即创建 ultralytics.models.yolo.detect.train.DetectionTrainer 类的对象作为trainer。此处仅读取数据集地址及其他参数
        if not args.get("resume"):  # 如果不是resume，手动将模型对象送入trainer
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

        self.trainer.hub_session = self.session  # attach optional HUB session
        self.trainer.train()
        # Update model and cfg after training
        if RANK in {-1, 0}:
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
        return self.metrics

    def tune(
        self,
        use_ray=False,
        iterations=10,
        *args,
        **kwargs,
    ):
        """
        Conducts hyperparameter tuning for the model, with an option to use Ray Tune.

        This method supports two modes of hyperparameter tuning: using Ray Tune or a custom tuning method.
        When Ray Tune is enabled, it leverages the 'run_ray_tune' function from the ultralytics.utils.tuner module.
        Otherwise, it uses the internal 'Tuner' class for tuning. The method combines default, overridden, and
        custom arguments to configure the tuning process.

        Args:
            use_ray (bool): If True, uses Ray Tune for hyperparameter tuning. Defaults to False.
            iterations (int): The number of tuning iterations to perform. Defaults to 10.
            *args (list): Variable length argument list for additional arguments.
            **kwargs (any): Arbitrary keyword arguments. These are combined with the model's overrides and defaults.

        Returns:
            (dict): A dictionary containing the results of the hyperparameter search.

        Raises:
            AssertionError: If the model is not a PyTorch model.
        """
        self._check_is_pytorch_model()
        if use_ray:
            from ultralytics.utils.tuner import run_ray_tune

            return run_ray_tune(self, max_samples=iterations, *args, **kwargs)
        else:
            from .tuner import Tuner

            custom = {}  # method defaults
            args = {**self.overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
            return Tuner(args=args, _callbacks=self.callbacks)(model=self, iterations=iterations)

    def _apply(self, fn) -> "Model":
        """Apply to(), cpu(), cuda(), half(), float() to model tensors that are not parameters or registered buffers."""
        self._check_is_pytorch_model()
        self = super()._apply(fn)  # noqa
        self.predictor = None  # reset predictor as device may have changed
        self.overrides["device"] = self.device  # was str(self.device) i.e. device(type='cuda', index=0) -> 'cuda:0'
        return self

    @property
    def names(self) -> list:
        """
        Retrieves the class names associated with the loaded model.

        This property returns the class names if they are defined in the model. It checks the class names for validity
        using the 'check_class_names' function from the ultralytics.nn.autobackend module.

        Returns:
            (list | None): The class names of the model if available, otherwise None.
        """
        from ultralytics.nn.autobackend import check_class_names

        if hasattr(self.model, "names"):
            return check_class_names(self.model.names)
        if not self.predictor:  # export formats will not have predictor defined until predict() is called
            self.predictor = self._smart_load("predictor")(overrides=self.overrides, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        return self.predictor.model.names

    @property
    def device(self) -> torch.device:
        """
        Retrieves the device on which the model's parameters are allocated.

        This property is used to determine whether the model's parameters are on CPU or GPU. It only applies to models
        that are instances of nn.Module.

        Returns:
            (torch.device | None): The device (CPU/GPU) of the model if it is a PyTorch model, otherwise None.
        """
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        """
        Retrieves the transformations applied to the input data of the loaded model.

        This property returns the transformations if they are defined in the model.

        Returns:
            (object | None): The transform object of the model if available, otherwise None.
        """
        return self.model.transforms if hasattr(self.model, "transforms") else None

    def add_callback(self, event: str, func) -> None:
        """
        Adds a callback function for a specified event.

        This method allows the user to register a custom callback function that is triggered on a specific event during
        model training or inference.

        Args:
            event (str): The name of the event to attach the callback to.
            func (callable): The callback function to be registered.

        Raises:
            ValueError: If the event name is not recognized.
        """
        self.callbacks[event].append(func)

    def clear_callback(self, event: str) -> None:
        """
        Clears all callback functions registered for a specified event.

        This method removes all custom and default callback functions associated with the given event.

        Args:
            event (str): The name of the event for which to clear the callbacks.

        Raises:
            ValueError: If the event name is not recognized.
        """
        self.callbacks[event] = []

    def reset_callbacks(self) -> None:
        """
        Resets all callbacks to their default functions.

        This method reinstates the default callback functions for all events, removing any custom callbacks that were
        added previously.
        """
        for event in callbacks.default_callbacks.keys():
            self.callbacks[event] = [callbacks.default_callbacks[event][0]]

    @staticmethod
    def _reset_ckpt_args(args: dict) -> dict:
        """Reset arguments when loading a PyTorch model."""
        include = {"imgsz", "data", "task", "single_cls"}  # only remember these arguments when loading a PyTorch model
        return {k: v for k, v in args.items() if k in include}

    # def __getattr__(self, attr):
    #    """Raises error if object has no requested attribute."""
    #    name = self.__class__.__name__
    #    raise AttributeError(f"'{name}' object has no attribute '{attr}'. See valid attributes below.\n{self.__doc__}")

    def _smart_load(self, key: str):
        """Load model/trainer/validator/predictor."""
        try:
            return self.task_map[self.task][key]  # 返回的是 ultralytics.models.yolo.detect.train.DetectionTrainer 这个类
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]  # get the function name.
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")
            ) from e

    @property
    def task_map(self) -> dict:
        """
        Map head to model, trainer, validator, and predictor classes.

        Returns:
            task_map (dict): The map of model task to mode classes.
        """
        raise NotImplementedError("Please provide task map for your model!")
