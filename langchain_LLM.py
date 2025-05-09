from typing import Any, Dict, Iterator, List, Optional
from openai import OpenAI
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    ChatMessage,
    HumanMessage
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
import time

# 由于LangChain的消息类型是HumanMessage、AIMessage等格式，与一般模型接收的字典格式不一样，因此我们需要先定义一个将LangChain的格式转为字典的函数。
def _convert_message_to_dict(message: BaseMessage) -> dict:
    """把LangChain的消息格式转为智谱支持的格式
    Args:
        message: The LangChain message.
    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {"content": message.content}
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict

# 继承自 LangChain 的 BaseChatModel 类
class siliconflowLLM(BaseChatModel):
    """自定义Zhipuai聊天模型。
    """
    model_name: str = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 3
    api_key: str | None = None


# 接下来我们实现_generate方法： _generate 方法，接收一系列消息及其他参数，并返回输出
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """通过调用智谱API从而响应输入。

        Args:
            messages: 由messages列表组成的prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为LLM提供回调的运行管理器
        """
        # 列表推导式 将 messages 的元素逐个转为智谱的格式
        messages = [_convert_message_to_dict(message) for message in messages]
        # 定义推理的开始时间
        start_time = time.time()
        # 调用 ZhipuAI 对处理消息
        response = OpenAI(api_key=self.api_key, base_url="https://api.siliconflow.cn/v1").chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages
        )
        # 计算运行时间 由现在时间 time.time() 减去 开始时间start_time得到
        time_in_seconds = time.time() - start_time
        # 将返回的消息封装并返回
        message = AIMessage(
            content=response.choices[0].message.content, # 响应的结果
            additional_kwargs={}, # 额外信息
            response_metadata={
                "time_in_seconds": round(time_in_seconds, 3), # 响应源数据 这里是运行时间 也可以添加其他信息
            },
            # 本次推理消耗的token
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens, # 输入token
                "output_tokens": response.usage.completion_tokens, # 输出token
                "total_tokens": response.usage.total_tokens, # 全部token
            },
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

# 接下来我们实现另一核心的方法_stream，之前注释的代码本次不再注释：_stream 方法， 接收一系列消息及其他参数，并以流式返回结果。
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """通过调用智谱API返回流式输出。

        Args:
            messages: 由messages列表组成的prompt
            stop: 在模型生成的回答中有该字符串列表中的元素则停止响应
            run_manager: 一个为LLM提供回调的运行管理器
        """
        messages = [_convert_message_to_dict(message) for message in messages]
        response = OpenAI().chat.completions.create(
            model=self.model_name,
            stream=True, # 将stream 设置为 True 返回的是迭代器，可以通过for循环取值
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
            stop=stop,
            messages=messages
        )
        start_time = time.time()
        # 使用for循环存取结果
        for res in response:
            if res.usage: # 如果 res.usage 存在则存储token使用情况
                usage_metadata = UsageMetadata(
                    {
                        "input_tokens": res.usage.prompt_tokens,
                        "output_tokens": res.usage.completion_tokens,
                        "total_tokens": res.usage.total_tokens,
                    }
                )
            # 封装每次返回的chunk
            chunk = ChatGenerationChunk(
                message=AIMessageChunk(content=res.choices[0].delta.content)
            )

            if run_manager:
                # This is optional in newer versions of LangChain
                # The on_llm_new_token will be called automatically
                run_manager.on_llm_new_token(res.choices[0].delta.content, chunk=chunk)
            # 使用yield返回 结果是一个生成器 同样可以使用for循环调用
            yield chunk
        time_in_sec = time.time() - start_time
        # Let's add some other information (e.g., response metadata)
        # 最终返回运行时间
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": round(time_in_sec, 3)}, usage_metadata=usage_metadata)
        )
        if run_manager:
            # This is optional in newer versions of LangChain
            # The on_llm_new_token will be called automatically
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

# 定义一下模型的描述方法：
    @property
    def _llm_type(self) -> str:
        """获取此聊天模型使用的语言模型类型。"""
        return self.model_name

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回一个标识参数的字典。

        该信息由LangChain回调系统使用，用于跟踪目的，使监视llm成为可能。
        """
        return {
            "model_name": self.model_name,
        }

