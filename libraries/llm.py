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
import pymysql
import sys


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


def get_csv (file_path,id):

    answer_generation_chain, ques_list = llm_pipeline(file_path)
    quest_db = ans_db = ''
    base_folder = 'static/output/'
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder+"QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            question += "&"
            quest_db += question
            answer = answer_generation_chain.run(question)
            print("Answer: ", answer)
            answer += "&"
            ans_db += answer
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, answer])
    
    cursor = connection.cursor()
    qry = 'UPDATE student_profile set question = "{}", answer = "{}" where stud_id = {}'
    qry = qry.format(quest_db,ans_db,id)
    print(qry)
    cursor.execute(qry)
    connection.commit()
    command = "select stud_email from student_profile where stud_id ={}"
    command = command.format(id)
    cursor.execute(command)
    email = cursor.fetchone()[0]
    send_mail(ques_list,email)
    return output_file


def send_mail(question_list,email):
    user = 'mithulcb@gmail.com'
    app_password = 'givd kvme dzcl bgyq ' # a token for gmail
    to=email
    content = ''
    subject = 'Questions for interview'
    for i in question_list:
        content += i +'\n'
    content += "\n\n**********Please Answer The Questions With Respect To Information In Your Resume**********"
    with yagmail.SMTP(user, app_password) as yag:
        yag.send(to, subject, content)
        print('Sent email successfully')

if __name__ == "__main__":

    connection = pymysql.connect(
    host='localhost',
    user='root',
    password='',
    db='Quiz_Database',
    use_unicode=True,
    charset="utf8")

    # path_to_res=sys.argv[1]
    # id = int(sys.argv[2])
    path_to_res = "CV/temp.pdf"
    id = 121

    cursor = connection.cursor()
    qry = 'UPDATE student_profile set flag = {} where stud_id = {}'
    qry = qry.format(1,id)
    cursor.execute(qry)
    connection.commit()


    get_csv(path_to_res,id)