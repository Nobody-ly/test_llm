import streamlit as st
# from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
# import sys
# sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中
from langchain_embedding import GuijiAIEmbeddings
from langchain_LLM import siliconflowLLM
from langchain_chroma import Chroma

API_KEY = "sk-foelcfvxqptsxbpjqjrjdhnymdiagvxodgsqyemusvstjlma"

# 该函数返回一个检索器，用于检索文档
def get_retriever():
    # 定义 Embeddings
    embedding = GuijiAIEmbeddings()
    # 向量数据库持久化路径
    persist_directory = './chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

# 该函数处理检索器返回的文本
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

# 该函数可以返回一个检索问答链
def get_qa_history_chain():
    retriever = get_retriever()
    # llm = siliconflowLLM(model_name="gpt-4o", temperature=0)
    llm = siliconflowLLM(model_name="Qwen/QwQ-32B", temperature=0., api_key=API_KEY)
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "你是一个问答任务的助手。 "
        "请使用检索到的上下文片段回答这个问题。 "
        "如果你不知道答案就说不知道。 "
        "请使用简洁的话语回答用户。"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(       # .from_messages()方法可以快速创建模板，但也可以不加
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs,
        ).assign(answer=qa_chain)
    return qa_history_chain

# 接受检索问答链、用户输入及聊天历史，并以流式返回该链输出
def gen_response(chain, input, chat_history):
    response = chain.stream({                   # .stream()方法可以流式处理输入，返回输出
        "input": input,
        "chat_history": chat_history
    })
    for res in response:                      # 遍历输出，此处response为生成器对象，每次迭代返回一个结果片段
        if "answer" in res.keys():           # 如果输出包含answer字段，则表示该结果片段为回答，直接输出
            yield res["answer"]             # 使用yield关键字逐步返回答案内容，这种方式允许函数在生成答案时保持流式输出

def main():
    st.markdown('### 🦜🔗 动手学大模型应用开发')
    # st.session_state可以存储用户与应用交互期间的状态与数据
    # 存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    # 建立容器 高度为500 px
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages: # 遍历对话历史
            with messages.chat_message(message[0]): # messages指在容器下显示，chat_message显示用户及ai头像
                st.write(message[1]) # 打印内容
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        # 显示当前用户输入
        with messages.chat_message("human"):
            st.write(prompt)
        # 生成回复
        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        # 流式输出
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        # 将输出存入st.session_state.messages
        st.session_state.messages.append(("ai", output))
