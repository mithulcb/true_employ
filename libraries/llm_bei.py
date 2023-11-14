from langchain.llms import CTransformers
from langchain.chains import QAGenerationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader
import csv
import os
from transformers import AutoModel
import yagmail


MBTI_TYPES = {
    "INTJ": "Imaginative and strategic thinkers, with a plan for everything.",
    "INTP": "Innovative inventors with an unquenchable thirst for knowledge.",
    "ENTJ": "Bold, imaginative and strong-willed leaders, always finding a way or making one.",
    "ENTP": "Smart and curious thinkers who cannot resist an intellectual challenge.",
    "INFJ": "Quiet and mystical, yet very inspiring and tireless idealists.",
    "INFP": "Poetic, kind and altruistic people, always eager to help a good cause.",
    "ENFJ": "Charismatic and inspiring leaders, able to mesmerize their listeners.",
    "ENFP": "Enthusiastic, creative and sociable free spirits, who can always find a reason to smile.",
    "ISTJ": "Practical and fact-minded individuals, whose reliability cannot be doubted.",
    "ISFJ": "Very dedicated and warm protectors, always ready to defend their loved ones.",
    "ESTJ": "Excellent administrators, unsurpassed at managing things or people.",
    "ESFJ": "Extraordinarily caring, social and popular people, always eager to help.",
    "ISTP": "Bold and practical experimenters, masters of all kinds of tools.",
    "ISFP": "Flexible and charming artists, always ready to explore and experience something new.",
    "ESTP": "Smart, energetic and very perceptive people, who truly enjoy living on the edge.",
    "ESFP": "Spontaneous, energetic and enthusiastic people life is never boring around them."
}



def load_llm():
    llm = CTransformers(
        model = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        model_type="mistral",
        max_new_tokens = 512,
        temperature = 0.3
    )
    return llm


def file_processing(file_path):

    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''

    for page in data:
        question_gen += page.page_content
        
    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = RecursiveCharacterTextSplitter(
        chunk_size = 300,
        chunk_overlap = 30
    )


    document_answer_gen = splitter_ans_gen.split_documents(
        document_ques_gen
    )

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):

    document_ques_gen, document_answer_gen = file_processing(file_path)



    llm_ques_gen_pipeline = load_llm()

    prompt_template = """
    You are an interviewer.
    Your objective is to assess a candidate's qualifications and suitability for the position by asking questions related to the candidate's resume. 
    You should frame questions that explore the candidate's skills, experiences, and achievements as outlined in their resume.
    You do this by asking questions about the text below which corresponds to the resume:

    ------------
    {text}
    ------------

    Create questions that will test the candidate on his knowledge and skills on his resume.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an interviewer.
    Your objective is to assess a candidate's qualifications and suitability for the position by asking questions related to the candidate's resume. 
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """
    )

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_ques_gen_pipeline, 
                                            chain_type = "refine", 
                                            verbose = True, 
                                            question_prompt=PROMPT_QUESTIONS, 
                                            refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = load_llm()

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                                chain_type="stuff", 
                                                retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list


def get_csv (file_path):
    answer_generation_chain, ques_list = llm_pipeline(file_path)
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])
    return output_file


def send_mail(question):
    user = ''
    app_password = '' # a token for gmail
    to=email
    content = ''
    subject = 'Questions'
    for i in question:
        content += i +'\n'
    with yagmail.SMTP(user, app_password) as yag:
        yag.send(to, subject, content)
        print('Sent email successfully')
