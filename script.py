import os
import random
import openai

from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from dotenv import load_dotenv

from constants import CHROMA_SETTINGS

load_dotenv()

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

if __name__ == "__main__":
     # Load document
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    persist_directory = os.environ.get("PERSIST_DIRECTORY")

    loader = PyPDFLoader("./COMP3221-W11-Lecture.pdf")
    documents = loader.load()

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # Create the OpenAI embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)

    # Expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})

    # Createa a chain to answer questions
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever, return_source_documents=True)

    random_index = random.randrange(len(texts))

    query = generate_quizzes(texts[random_index])
    print ('********************\n\n')
    print('GENERATED QUESTION:', query, '\n\n')
    result = qa({"query": query})
    print ('********************\n')
    user_answer = input("Enter your answer: ")
    print ('\n********* Generating Feedback ***********\n')
    feedback = evaluate_quizzes(query, result, user_answer)
    print("FEEDBACK:", feedback)

    print ('\n\n\n********* Complete Answer ***********\n')  
    print('CORRECT ANSWER:', result['result'])
