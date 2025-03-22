"""
模型接口模块 - 支持多种大语言模型提供商
"""

import os
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
import requests
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    """聊天消息模型"""
    role: str = Field(..., description="消息角色: system, user, assistant")
    content: str = Field(..., description="消息内容")


class ModelInterface(ABC):
    """模型接口基类"""

    @abstractmethod
    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """生成回复"""
        pass

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        pass

    @abstractmethod
    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass


class OllamaModel(ModelInterface):
    """Ollama 本地模型接口"""

    def __init__(self, model_name: str = "llama3", base_url: str = "http://localhost:11434"):
        """
        初始化Ollama模型

        Args:
            model_name: Ollama模型名称
            base_url: Ollama服务URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/chat"
        self.system_prompt = "你是一个知识丰富的智能助手。请根据用户问题提供准确、有用的回答。"
        self.temperature = 0.7
        self.__model_type__ = "ollama"

        # 检查模型可用性
        try:
            self._check_model_availability()
            logger.info(f"Ollama模型'{model_name}'初始化成功")
        except Exception as e:
            logger.warning(f"Ollama模型'{model_name}'初始化警告: {str(e)}")

    def _check_model_availability(self) -> None:
        """检查模型是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])

            # 检查模型是否在列表中
            model_exists = any(model.get("name") == self.model_name for model in models)

            if not model_exists:
                logger.warning(
                    f"模型'{self.model_name}'在Ollama中不可用，可能需要先使用'ollama pull {self.model_name}'下载")
        except Exception as e:
            raise ConnectionError(f"无法连接到Ollama服务({self.base_url}): {str(e)}")

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        self.temperature = max(0.0, min(1.0, temperature))

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        生成回复

        Args:
            query: 用户问题
            chat_history: 聊天历史

        Returns:
            模型回复文本
        """
        if chat_history is None:
            chat_history = []

        messages = [{"role": "system", "content": self.system_prompt}]

        for message in chat_history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["content"]})
            else:
                messages.append({"role": "assistant", "content": message["content"]})

        messages.append({"role": "user", "content": query})

        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": 2048  # 设置合理的回复长度上限
                    }
                },
                timeout=60  # 设置超时时间
            )
            response.raise_for_status()
            elapsed_time = time.time() - start_time

            result = response.json()["message"]["content"]
            logger.info(f"Ollama响应时间: {elapsed_time:.2f}秒, 返回字符数: {len(result)}")
            return result
        except requests.exceptions.Timeout:
            logger.error("Ollama请求超时")
            return "抱歉，模型响应超时。请稍后再试或尝试简化您的问题。"
        except requests.exceptions.ConnectionError:
            logger.error(f"无法连接到Ollama服务({self.base_url})")
            return f"错误: 无法连接到Ollama服务。请确保Ollama服务正在运行并可在{self.base_url}访问。"
        except Exception as e:
            logger.error(f"Ollama请求错误: {str(e)}")
            return f"错误: 调用Ollama服务时发生问题 - {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            response = requests.get(f"{self.base_url}/api/show", params={"name": self.model_name})
            response.raise_for_status()
            model_info = response.json()
            return {
                "name": self.model_name,
                "provider": "Ollama",
                "parameters": model_info.get("parameters", "未知"),
                "type": "本地模型",
                "context_window": model_info.get("context_length", "未知")
            }
        except Exception:
            return {
                "name": self.model_name,
                "provider": "Ollama",
                "type": "本地模型",
                "status": "无法获取详细信息"
            }


class MoonshotModel(ModelInterface):
    """Moonshot AI API接口"""

    def __init__(self, api_key: str, model_name: str = "moonshot-v1-8k"):
        """
        初始化Moonshot模型

        Args:
            api_key: Moonshot API密钥
            model_name: 模型名称
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = "https://api.moonshot.cn/v1/chat/completions"
        self.system_prompt = "你是一个知识丰富的智能助手。请根据用户问题提供准确、有用的回答。"
        self.temperature = 0.7
        self.__model_type__ = "moonshot"

        # 模型配置映射
        self.model_config = {
            "moonshot-v1-8k": {"context_window": 8000, "type": "通用对话"},
            "moonshot-v1-32k": {"context_window": 32000, "type": "通用对话"},
            "moonshot-v1-128k": {"context_window": 128000, "type": "通用对话"}
        }

        # 验证API密钥
        if not api_key:
            logger.warning("未提供Moonshot API密钥，API调用可能会失败")

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        self.temperature = max(0.0, min(1.0, temperature))

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        生成回复

        Args:
            query: 用户问题
            chat_history: 聊天历史

        Returns:
            模型回复文本
        """
        if chat_history is None:
            chat_history = []

        messages = [{"role": "system", "content": self.system_prompt}]

        for message in chat_history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["content"]})
            else:
                messages.append({"role": "assistant", "content": message["content"]})

        messages.append({"role": "user", "content": query})

        try:
            if not self.api_key:
                return "错误: 未配置Moonshot API密钥。请在配置文件或环境变量中设置MOONSHOT_API_KEY。"

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": 0.7,  # 添加top_p参数提高回复质量
                    "max_tokens": 4096  # 设置最大令牌数
                },
                timeout=60  # 设置超时时间
            )
            response.raise_for_status()
            elapsed_time = time.time() - start_time

            result = response.json()["choices"][0]["message"]["content"]
            logger.info(f"Moonshot响应时间: {elapsed_time:.2f}秒, 返回字符数: {len(result)}")
            return result
        except requests.exceptions.Timeout:
            logger.error("Moonshot API请求超时")
            return "抱歉，Moonshot API响应超时。请稍后再试或尝试简化您的问题。"
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else "未知"
            error_detail = ""

            try:
                error_json = e.response.json()
                error_detail = error_json.get("error", {}).get("message", "")
            except:
                pass

            logger.error(f"Moonshot API错误 (状态码: {status_code}): {error_detail}")
            return f"错误: Moonshot API返回错误 (状态码: {status_code}) - {error_detail}"
        except Exception as e:
            logger.error(f"Moonshot请求错误: {str(e)}")
            return f"错误: 调用Moonshot API时发生问题 - {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        model_config = self.model_config.get(self.model_name, {})
        return {
            "name": self.model_name,
            "provider": "Moonshot AI",
            "type": model_config.get("type", "通用对话"),
            "context_window": model_config.get("context_window", "未知")
        }


class LMStudioModel(ModelInterface):
    """LM Studio 本地模型接口"""

    def __init__(self, base_url: str = "http://localhost:1234/v1", model_name: str = "Local Model"):
        """
        初始化LM Studio模型

        Args:
            base_url: LM Studio服务URL
            model_name: 模型名称标识符
        """
        self.base_url = base_url
        self.model_name = model_name
        self.api_url = f"{base_url}/chat/completions"
        self.system_prompt = "你是一个知识丰富的智能助手。请根据用户问题提供准确、有用的回答。"
        self.temperature = 0.7
        self.__model_type__ = "lmstudio"

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        self.temperature = max(0.0, min(1.0, temperature))

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        生成回复

        Args:
            query: 用户问题
            chat_history: 聊天历史

        Returns:
            模型回复文本
        """
        if chat_history is None:
            chat_history = []

        messages = [{"role": "system", "content": self.system_prompt}]

        for message in chat_history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["content"]})
            else:
                messages.append({"role": "assistant", "content": message["content"]})

        messages.append({"role": "user", "content": query})

        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json={
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 2048,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            elapsed_time = time.time() - start_time

            result = response.json()["choices"][0]["message"]["content"]
            logger.info(f"LM Studio响应时间: {elapsed_time:.2f}秒, 返回字符数: {len(result)}")
            return result
        except requests.exceptions.Timeout:
            return "抱歉，LM Studio模型响应超时。请稍后再试或检查服务是否正常运行。"
        except Exception as e:
            logger.error(f"LM Studio请求错误: {str(e)}")
            return f"错误: 无法连接到LM Studio服务 - {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.model_name,
            "provider": "LM Studio",
            "type": "本地模型",
            "context_window": "未知"  # LM Studio API不直接提供上下文窗口大小
        }


class OpenAICompatibleModel(ModelInterface):
    """OpenAI兼容API接口 (支持Claude, DeepSeek等兼容OpenAI API的服务)"""

    def __init__(self,
                 api_key: str,
                 base_url: str,
                 model_name: str,
                 provider_name: str = "Custom API"):
        """
        初始化OpenAI兼容模型

        Args:
            api_key: API密钥
            base_url: API基础URL
            model_name: 模型名称
            provider_name: 提供商名称
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self.provider_name = provider_name
        self.api_url = f"{base_url}/chat/completions"
        self.system_prompt = "你是一个知识丰富的智能助手。请根据用户问题提供准确、有用的回答。"
        self.temperature = 0.7
        self.__model_type__ = "openai_compatible"

    def set_system_prompt(self, prompt: str) -> None:
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_temperature(self, temperature: float) -> None:
        """设置温度参数"""
        self.temperature = max(0.0, min(1.0, temperature))

    def generate_response(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        生成回复

        Args:
            query: 用户问题
            chat_history: 聊天历史

        Returns:
            模型回复文本
        """
        if chat_history is None:
            chat_history = []

        messages = [{"role": "system", "content": self.system_prompt}]

        for message in chat_history:
            if message["role"] == "user":
                messages.append({"role": "user", "content": message["content"]})
            else:
                messages.append({"role": "assistant", "content": message["content"]})

        messages.append({"role": "user", "content": query})

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 2048
                },
                timeout=60
            )
            response.raise_for_status()
            elapsed_time = time.time() - start_time

            result = response.json()["choices"][0]["message"]["content"]
            logger.info(f"{self.provider_name}响应时间: {elapsed_time:.2f}秒, 返回字符数: {len(result)}")
            return result
        except Exception as e:
            logger.error(f"{self.provider_name}请求错误: {str(e)}")
            return f"错误: 调用{self.provider_name} API时发生问题 - {str(e)}"

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "name": self.model_name,
            "provider": self.provider_name,
            "type": "API模型"
        }


def get_model(model_type: str, config_data: Optional[Dict[str, Any]] = None) -> ModelInterface:
    """
    获取指定类型的模型实例

    Args:
        model_type: 模型类型标识符
        config_data: 模型配置数据

    Returns:
        模型接口实例

    Raises:
        ValueError: 不支持的模型类型
    """
    import config as app_config

    if config_data is None:
        if model_type in app_config.MODEL_CONFIG:
            config_data = app_config.MODEL_CONFIG[model_type]
        else:
            raise ValueError(f"配置中不存在模型类型: {model_type}")

    if model_type == "ollama":
        return OllamaModel(
            model_name=config_data.get("model_name", "llama3"),
            base_url=config_data.get("base_url", "http://localhost:11434")
        )
    elif model_type == "moonshot":
        return MoonshotModel(
            api_key=config_data.get("api_key", os.environ.get("MOONSHOT_API_KEY", "")),
            model_name=config_data.get("model_name", "moonshot-v1-8k")
        )
    elif model_type == "lmstudio":
        return LMStudioModel(
            base_url=config_data.get("base_url", "http://localhost:1234/v1"),
            model_name=config_data.get("model_name", "本地模型")
        )
    elif model_type == "deepseek":
        return OpenAICompatibleModel(
            api_key=config_data.get("api_key", os.environ.get("DEEPSEEK_API_KEY", "")),
            base_url=config_data.get("base_url", "https://api.deepseek.com/v1"),
            model_name=config_data.get("model_name", "deepseek-chat"),
            provider_name="DeepSeek"
        )
    elif model_type == "custom_openai":
        return OpenAICompatibleModel(
            api_key=config_data.get("api_key", ""),
            base_url=config_data.get("base_url", ""),
            model_name=config_data.get("model_name", ""),
            provider_name=config_data.get("provider_name", "自定义API")
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")