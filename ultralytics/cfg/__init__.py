# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_YAML,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    checks,
    colorstr,
    deprecation_warn,
    yaml_load,
    yaml_print,
)

# Define valid tasks and modes
MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolov8n.pt",
    "segment": "yolov8n-seg.pt",
    "classify": "yolov8n-cls.pt",
    "pose": "yolov8n-pose.pt",
    "obb": "yolov8n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
CLI_HELP_MSG = f"""
    Arguments received: {str(['yolo'] + ARGV[1:])}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    5. Explore your datasets using semantic search and SQL with a simple GUI powered by Ultralytics Explorer API
        yolo explorer
    
    6. Streamlit real-time object detection on your webcam with Ultralytics YOLOv8
        yolo streamlit-predict
        
    7. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Define keys for arg type checks
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {  # integer-only arguments
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {  # boolean-only arguments
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration object to be converted to a dictionary. This may be a
            path to a configuration file, a dictionary, or a SimpleNamespace object.

    Returns:
        (Dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict('config.yaml')

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1='value1', param2='value2')
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({'param1': 'value1', 'param2': 'value2'})

    Notes:
        - If `cfg` is a path or a string, it will be loaded as YAML and converted to a dictionary.
        - If `cfg` is a SimpleNamespace object, it will be converted to a dictionary using `vars()`.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data source.
        overrides (Dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged training arguments.

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric `project`
          and `name` to strings and validating configuration keys and values.

    Examples:
        Load default configuration:
        >>> from ultralytics import get_cfg
        >>> config = get_cfg()

        Load from a custom file with overrides:
        >>> config = get_cfg('path/to/config.yaml', overrides={'epochs': 50, 'batch_size': 16})
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg, hard=True):
    """
    Checks configuration argument types and values for the Ultralytics library, ensuring correctness and converting them
    if necessary.

    Args:
        cfg (Dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        Validate a configuration with a mix of valid and invalid values:
        >>> config = {
        ...     'epochs': 50,         # valid integer
        ...     'lr0': 0.01,         # valid float
        ...     'momentum': 1.2,     # invalid float (out of 0.0-1.0 range)
        ...     'save': 'true',      # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key and retained other values
    """
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                cfg[k] = float(v)
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                    )
                cfg[k] = bool(v)


def get_save_dir(args, name=None):
    """
    返回用于保存输出的目录路径，该路径来自参数或默认设置。
    参数：
        args（SimpleNamespace）：包含配置的命名空间对象，如 'project'、'name'、'task'、'mode' 和 'save_dir'。
        name（str | None）：输出目录的可选名称。如果未提供，则默认为 'args.name' 或 'args.mode'。
    返回：
        (Path)：保存输出的目录路径。

    Examples:
        Generate a save directory using provided arguments
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project='my_project', task='detect', mode='train', exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        my_project/detect/train
    """

    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)


def _handle_deprecation(custom):
    """
    Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (Dict): Configuration dictionary potentially containing deprecated keys.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}
    """

    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")

    return custom


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    Check for key alignment between custom and base configuration dictionaries, handling deprecated keys and providing
    informative error messages for mismatched keys.

    Args:
        base (Dict): The base configuration dictionary containing valid keys.
        custom (Dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function. Default is None.

    Raises:
        SystemExit: Terminates the program execution if mismatched keys are found.

    Notes:
        - The function suggests corrections for mismatched keys based on similarity to valid keys.
        - Deprecated keys in the custom configuration are automatically replaced with their updated equivalents.
        - Detailed error messages are printed for each mismatched key to help users identify and correct their custom
          configurations.

    Examples:
        >>> base_cfg = {'epochs': 50, 'lr0': 0.01, 'batch_size': 16}
        >>> custom_cfg = {'epoch': 100, 'lr': 0.02, 'batch_size': 32}

        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     # Handle the error or correct the configuration
        ...     pass
    """
    custom = _handle_deprecation(custom)
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' in a list of strings.

    Args:
        args (List[str]): A list of strings where each element represents an argument.

    Returns:
        (List[str]): A list of strings where the arguments around isolated '=' are merged.

    Examples:
        Merge arguments where equals sign is separated:
        >>> args = ["arg1", "=", "value"]
        >>> merge_equals_args(args)
        ["arg1=value"]

        Merge arguments where equals sign is at the end of the first argument:
        >>> args = ["arg1=", "value"]
        >>> merge_equals_args(args)
        ["arg1=value"]

        Merge arguments where equals sign is at the beginning of the second argument:
        >>> args = ["arg1", "=value"]
        >>> merge_equals_args(args)
        ["arg1=value"]
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            del args[i + 1]
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            del args[i + 1]
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args


def handle_yolo_hub(args: List[str]) -> None:
    """
    Handle Ultralytics HUB command-line interface (CLI) commands.

    This function processes Ultralytics HUB CLI commands such as login and logout. It should be called when executing a
    script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments.

    Examples:
        ```bash
        yolo hub login YOUR_API_KEY
        ```
    """
    from ultralytics import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""
        # Log in to Ultralytics HUB using the provided API key
        hub.login(key)
    elif args[0] == "logout":
        # Log out from Ultralytics HUB
        hub.logout()


def handle_yolo_settings(args: List[str]) -> None:
    """
    Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset. It should be called when executing a script with
    arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Examples:
        Reset YOLO settings:
        >>> yolo settings reset

    Notes:
        For more information on handling YOLO settings, visit:
        https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"  # help URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_YAML.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info("Settings reset successfully")  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        LOGGER.info(f"💡 Learn about settings at {url}")
        yaml_print(SETTINGS_YAML)  # print the current settings
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")


def handle_explorer():
    """
    Open the Ultralytics Explorer GUI for dataset exploration and analysis.

    This function launches a graphical user interface that provides tools for interacting with and analyzing datasets
    using the Ultralytics Explorer API.

    Examples:
        Start the Ultralytics Explorer:
        >>> handle_explorer()
    """
    checks.check_requirements("streamlit>=1.29.0")
    LOGGER.info("💡 Loading Explorer dashboard...")
    subprocess.run(["streamlit", "run", ROOT / "data/explorer/gui/dash.py", "--server.maxMessageSize", "2048"])


def handle_streamlit_inference():
    """
    Open the Ultralytics Live Inference streamlit app for real-time object detection.

    This function initializes and runs a Streamlit application designed for performing live object detection using
    Ultralytics models.

    References:
        - Streamlit documentation: https://docs.streamlit.io/
        - Ultralytics: https://docs.ultralytics.com

    Examples:
        To run the live inference Streamlit app, execute:
        >>> handle_streamlit_inference()
    """
    checks.check_requirements("streamlit>=1.29.0")
    LOGGER.info("💡 Loading Ultralytics Live Inference app...")
    subprocess.run(["streamlit", "run", ROOT / "solutions/streamlit_inference.py", "--server.headless", "true"])


def parse_key_value_pair(pair):
    """
    Parse a 'key=value' pair and return the key and value.

    Args:
        pair (str): The 'key=value' string to be parsed.

    Returns:
        (tuple[str, str]): A tuple containing the key and value as separate strings.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolov8n.pt")
        >>> key
        'model'
        >>> value
        'yolov8n.pt
    """
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """
    Convert a string representation of a value into its appropriate Python type (int, float, bool, None, etc.).

    Args:
        v (str): String representation of the value to be converted.

    Returns:
        (Any): The converted value, which can be of type int, float, bool, None, or the original string if no conversion
            is applicable.

    Examples:
        Convert a string to various types:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> smart_value("None")
        None
        >>> smart_value("some_string")
        'some_string'
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        with contextlib.suppress(Exception):
            return eval(v)
        return v


def entrypoint(debug=""):
    """
    Ultralytics entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the Ultralytics CLI, parsing command-line arguments and
    executing the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str, optional): Space-separated string of command-line arguments for debugging purposes.

    Examples:
        Train a detection model for 10 epochs with an initial learning_rate of 0.01:
        >>> entrypoint("train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01")

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        >>> entrypoint("predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        Validate a pretrained detection model at batch-size 1 and image size 640:
        >>> entrypoint("val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640")

    Notes:
        - For a list of all available commands and their arguments, see the provided help messages and the Ultralytics
          documentation at https://docs.ultralytics.com.
        - If no arguments are passed, the function will display the usage help message.
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: yaml_print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "explorer": lambda: handle_explorer(),
        "streamlit-predict": lambda: handle_streamlit_inference(),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 'mode' argument is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model", DEFAULT_CFG.model)
    if model is None:
        model = "yolov8n.pt"
        LOGGER.warning(f"WARNING ⚠️ 'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics import RTDETR

        model = RTDETR(model)  # no task argument
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING ⚠️ conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = DEFAULT_CFG.source or ASSETS
        LOGGER.warning(f"WARNING ⚠️ 'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """
    Copy and create a new default configuration file with '_copy' appended to its name, providing a usage example.

    This function duplicates the existing default configuration file and appends '_copy' to its name in the current
    working directory.

    Examples:
        Copy the default configuration file and use it in a YOLO command:
        >>> copy_default_cfg()
        >>> # Example YOLO command with this new custom cfg:
        >>> # yolo cfg='default_copy.yaml' imgsz=320 batch=8
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolov8n.pt')
    entrypoint(debug="")
