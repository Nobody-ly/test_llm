from typing import List
from langchain_core.embeddings import Embeddings

# 定义一个继承自 Embeddings 类的自定义 Embeddings 类
class GuijiAIEmbeddings(Embeddings):
    """`Zhipuai Embeddings` embedding models."""
    def __init__(self):
        """
        实例化ZhipuAI为values["client"]

        Args:

            values (Dict): 包含配置信息的字典，必须包含 client 的字段.
        Returns:

            values (Dict): 包含配置信息的字典。如果环境中有zhipuai库，则将返回实例化的ZhipuAI类；否则将报错 'ModuleNotFoundError: No module named 'zhipuai''.
        """
        from openai import OpenAI
        API_KEY = "sk-foelcfvxqptsxbpjqjrjdhnymdiagvxodgsqyemusvstjlma"
        self.client = OpenAI(api_key=API_KEY, base_url="https://api.siliconflow.cn/v1")

    # embed_documents是对字符串列表（List[str]）计算embedding的方法
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        生成输入文本列表的 embedding.
        Args:
            texts (List[str]): 要生成 embedding 的文本列表.

        Returns:
            List[List[float]]: 输入列表中每个文档的 embedding 列表。每个 embedding 都表示为一个浮点值列表。
        """
        embeddings = self.client.embeddings.create(
            model="BAAI/bge-m3",
            input=texts
        )
        return [embeddings.embedding for embeddings in embeddings.data]

    # embed_query是对单个文本（str）计算embedding的方法
    def embed_query(self, text: str) -> List[float]:
        """
        生成输入文本的 embedding.

        Args:
            texts (str): 要生成 embedding 的文本.

        Return:
            embeddings (List[float]): 输入文本的 embedding，一个浮点数值列表.
        """

        return self.embed_documents([text])[0]

