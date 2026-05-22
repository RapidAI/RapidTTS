# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import platform
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Tuple

from onnxruntime import get_available_providers, get_device

from ...logger import logger


class OrtBackend(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    DML = "dml"
    CANN = "cann"
    COREML = "coreml"


class EP(Enum):
    CPU_EP = "CPUExecutionProvider"
    CUDA_EP = "CUDAExecutionProvider"
    DIRECTML_EP = "DmlExecutionProvider"
    CANN_EP = "CANNExecutionProvider"
    COREML_EP = "CoreMLExecutionProvider"


class IExecutionProvider(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def print_log(self, log_list: List[str]):
        for log_info in log_list:
            logger.warning(log_info)


class CPUExecutionProvider(IExecutionProvider):
    def __init__(self, cfg, had_providers: List[str]):
        self.cfg = cfg
        self.had_providers = had_providers
        self.default_provider = had_providers[0]

    def is_available(self) -> bool:
        return True

    def get_config(self) -> Dict[str, Any]:
        return {}

    @property
    def name(self) -> str:
        return EP.CPU_EP.value


class CUDAExecutionProvider(IExecutionProvider):
    def __init__(self, cfg, had_providers: List[str]):
        self.cfg = cfg
        self.had_providers = had_providers
        self.default_provider = had_providers[0]

    def is_available(self) -> bool:
        CUDA_EP = EP.CUDA_EP.value
        if get_device() == "GPU" and CUDA_EP in get_available_providers():
            return True

        logger.warning(
            f"{CUDA_EP} is not in available providers ({self.had_providers})."
        )
        install_instructions = [
            f"If you want to use {CUDA_EP} acceleration, you must do:"
            "(For reference only) If you want to use GPU acceleration, you must do:",
            "First, uninstall all onnxruntime packages in current environment.",
            "Second, install onnxruntime-gpu by `pip install onnxruntime-gpu`.",
            "Note the onnxruntime-gpu version must match your cuda and cudnn version.",
            "You can refer this link: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html",
            f"Third, ensure {CUDA_EP} is in available providers list. e.g. ['CUDAExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def get_config(self) -> Dict[str, Any]:
        return self.cfg[OrtBackend.CUDA.value] or {}

    @property
    def name(self) -> str:
        return EP.CUDA_EP.value


class DMLExecutionProvider(IExecutionProvider):
    def __init__(self, cfg, had_providers: List[str]):
        self.cfg = cfg
        self.had_providers = had_providers
        self.default_provider = had_providers[0]

    def is_available(self) -> bool:
        cur_os = platform.system()
        if cur_os != "Windows":
            logger.warning(
                f"DirectML is only supported in Windows OS. The current OS is {cur_os}.",
            )
            return False

        window_build_number_str = platform.version().split(".")[-1]
        window_build_number = (
            int(window_build_number_str) if window_build_number_str.isdigit() else 0
        )
        if window_build_number < 18362:
            logger.warning(
                f"DirectML is only supported in Windows 10 Build 18362 and above OS. The current Windows Build is {window_build_number}.",
            )
            return False

        DML_EP = EP.DIRECTML_EP.value
        if DML_EP in self.had_providers:
            return True

        logger.warning(
            f"{DML_EP} is not in available providers ({self.had_providers})."
        )
        install_instructions = [
            "If you want to use DirectML acceleration, you must do:",
            "First, uninstall all onnxruntime packages in current environment.",
            "Second, install onnxruntime-directml by `pip install onnxruntime-directml`",
            f"Third, ensure {DML_EP} is in available providers list. e.g. ['DmlExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def get_config(self) -> Dict[str, Any]:
        return self.cfg[OrtBackend.DML.value] or {}

    @property
    def name(self) -> str:
        return EP.DIRECTML_EP.value


class CANNExecutionProvider(IExecutionProvider):
    def __init__(self, cfg, had_providers: List[str]):
        self.cfg = cfg
        self.had_providers = had_providers
        self.default_provider = had_providers[0]

    def is_available(self) -> bool:
        CANN_EP = EP.CANN_EP.value
        if CANN_EP in self.had_providers:
            return True

        logger.warning(
            f"{CANN_EP} is not in available providers ({self.had_providers})."
        )
        install_instructions = [
            "If you want to use CANN acceleration, you must do:",
            "First, ensure you have installed Huawei Ascend software stack.",
            "Second, install onnxruntime with CANN support by following the instructions at:",
            "\thttps://onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html",
            f"Third, ensure {CANN_EP} is in available providers list. e.g. ['CANNExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def get_config(self) -> Dict[str, Any]:
        return self.cfg[OrtBackend.CANN.value] or {}

    @property
    def name(self) -> str:
        return EP.CANN_EP.value


class CoreMLExecutionProvider(IExecutionProvider):
    def __init__(self, cfg, had_providers: List[str]):
        self.cfg = cfg
        self.had_providers = had_providers
        self.default_provider = had_providers[0]

    def is_available(self) -> bool:
        cur_os = platform.system()
        if cur_os != "Darwin":
            logger.warning(
                f"CoreML is only supported in macOS/iOS. The current OS is {cur_os}.",
            )
            return False

        COREML_EP = EP.COREML_EP.value
        if COREML_EP in self.had_providers:
            return True

        logger.warning(
            f"{COREML_EP} is not in available providers ({self.had_providers})."
        )
        install_instructions = [
            "The standard onnxruntime package for macOS includes CoreML support.",
            f"Ensure {COREML_EP} is in available providers list. e.g. ['CoreMLExecutionProvider', 'CPUExecutionProvider']",
        ]
        self.print_log(install_instructions)
        return False

    def get_config(self) -> Dict[str, Any]:
        return self.cfg[OrtBackend.COREML.value] or {}

    @property
    def name(self) -> str:
        return EP.COREML_EP.value


class EPFactory:
    _EP_CLASSES = {
        OrtBackend.CUDA.value: CUDAExecutionProvider,
        OrtBackend.DML.value: DMLExecutionProvider,
        OrtBackend.CANN.value: CANNExecutionProvider,
        OrtBackend.COREML.value: CoreMLExecutionProvider,
    }

    @staticmethod
    def get_ep_list(cfg, had_providers: List[str], device: str) -> Any:
        default = [(EP.CPU_EP.value)]

        ep_class = EPFactory._EP_CLASSES.get(device)
        if not ep_class:
            return default

        ep = ep_class(cfg, had_providers)
        if ep.is_available():
            return [(ep.name, ep.get_config())]
        return default


def get_ep_list(cfg: Dict[str, Any], device: str) -> List[Tuple[str, Dict[str, Any]]]:
    return EPFactory.get_ep_list(cfg, get_available_providers(), device)
