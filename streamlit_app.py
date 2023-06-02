import os
import openai
import random
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)

from constants import CHROMA_SETTINGS

load_dotenv() # Load variables from .env file

def generate_quizzes(document):
    # Construct the prompt for the GPT model
    prompt = f"Generate 1 quiz question based on this text, which has 4 answers and 1 of them is correct. Do not include the answer: {document}"

    # Call the OpenAI API Chat Completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You are professor in Distributed System and Blockchain. You will create a quiz questions for students of the lecture on blockchains so that students can understand more about the subject.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=2400,
        n=1,
        stop=None,
        temperature=0.7
    )

    reply = response["choices"][0]["message"]["content"]
    return reply

def evaluate_quizzes(query, result, user_answer):
    # Construct prompt for the GPT model
    prompt = f"Give feedback and correct answer to this student as if you are quizzing on this question, telling them if they are correct, why the answer is correct and how to improve if applicable: {query} 'correct answer': '{result['result']}', student answer: {user_answer}"

    # Call the OpenAI API Chat Completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [
            {
                "role": "system",
                "content": "You are a teacher. I need you to help grade exams."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=2400,
        n=1,
        stop=None,
        temperature=0.7
    )

    reply = response["choices"][0]["message"]["content"]
    return reply


class Message:
    ai_icon = "./img/robot.png"

    def __init__(self, label: str, expanded: bool = True):
        self.label = label
        self.expanded = expanded

    def __enter__(self):
        message_area, icon_area = st.columns([10, 1])
        icon_area.image(self.ai_icon, caption="QuizMeGPT")

        self.expander = message_area.expander(label=self.label, expanded=self.expanded)

        return self

    def __exit__(self, ex_type, ex_value, trace):
        pass

    def write(self, content):
        self.expander.markdown(content)

def quizflow(texts, openai_api_key):    
    persist_directory = os.environ.get("PERSIST_DIRECTORY")
    
    # Select which embeddings we want to use
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Create the vector store to use as the index
    # db = Chroma.from_documents(texts, embeddings)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

    # Create a chain to answer questions
    
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)
    # # Generate a question
    # while True:
    #     random_index = random.randrange(len(texts))
    #     text = texts[random_index]

    #     query = generate_quizzes(text)
    #     result = qa({"query": query})

    #     with Message(label="Question") as m:
    #         m.write("### Question")
    #         m.write(query)
        
    #     user_answer = st.text_input("Write your answer here:")
    #     submit_button = st.button("Submit")

    #     if submit_button:
    #         feedback = evaluate_quizzes(query, result, user_answer)
    #         print("Feedback here:", feedback)
    #         provide_feedback(feedback, result)

    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False

    if not st.session_state['initialized']:
        random_index = random.randrange(len(texts))
        text = texts[random_index]
        st.session_state['query'] = generate_quizzes(text)
        st.session_state['result'] = qa({"query": st.session_state['query']})
        st.session_state['initialized'] = True

    with Message(label="Question") as m:
        m.write("### Question")
        m.write(st.session_state['query'])
        
    user_answer = st.text_input("Write your answer here:")
    submit_button = st.button("Submit")

    if submit_button:
        feedback = evaluate_quizzes(st.session_state['query'], st.session_state['result'], user_answer)
        print("Feedback here:", feedback)
        provide_feedback(feedback, st.session_state['result'])

        st.session_state['initialized'] = False

def provide_feedback(feedback, result):
    with Message(label="Feedback") as m:
        m.write("### Here's how you did!")
        m.write(feedback)
    
    with Message(label="Result") as mr:
        mr.write("### Correct Answer")
        mr.write(result["result"])
    
    new_question_button = st.button("New Question")
    print("New question button:", new_question_button)
    if new_question_button:
        quizflow(texts, openai_api_key)

if __name__ == "__main__":
    st.set_page_config(
        initial_sidebar_state="expanded",
        page_title="QuizMeGPT Streamlit",
        layout="centered",
    )

    with st.sidebar:
        openai_api_key = st.text_input('Your OpenAI API KEY', type="password")

    st.title("QuizMeGPT")

    if openai_api_key:
        loader = PyPDFLoader("./COMP3221-W11-Lecture.pdf")
        documents = loader.load()

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        quizflow(texts, openai_api_key)