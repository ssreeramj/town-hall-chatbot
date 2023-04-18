import os
from dotenv import load_dotenv

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

import gradio as gr
import time

load_dotenv()  # take environment variables from .env.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# load the trained model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

docsearch = FAISS.load_local("base-20230418_1930-index", embeddings)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=1024)

chain = load_qa_chain(llm, chain_type="map_rerank", verbose=False)

# Chatbot UI
with gr.Blocks() as demo:
    gr.Markdown("## Tiger Analytics Town Hall Q1 2023!!")
    chatbot = gr.Chatbot(label="Tiger Bot").style(height=400)

    with gr.Row():
        with gr.Column(scale=0.90):
            msg = gr.Textbox(
                show_label=False,
                placeholder="What do you want to know about the town hall?",
            ).style(container=False)
        with gr.Column(scale=0.10, min_width=0):
            btn = gr.Button("Send")

    clear = gr.Button("Clear")

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        # get user query
        query = history[-1][0]

        # get relevent documents through similarity search
        relevent_docs = docsearch.similarity_search(query=query, k=4)

        # pass the relevant docs to the chat model to generate the final answer.
        bot_message = chain(
            {"input_documents": relevent_docs, "question": query},
            return_only_outputs=True,
        )["output_text"].strip()

        history[-1][1] = bot_message
        time.sleep(1)
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

    gr.Markdown("## Some Example Questions")
    gr.Examples(
        [
            "What are some new companies that got involved with us?",
            "What were the disadvantages of working remotely?",
        ],
        [msg],
    )

demo.launch(auth=("tiger", "t!ger"))
