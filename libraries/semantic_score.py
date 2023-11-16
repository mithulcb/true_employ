import tensorflow as tf       
import tensorflow_hub as hub  
from numpy import dot                                           
from numpy.linalg import norm      
import numpy as np
import pandas as pd

def embed(input):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" 
    model = hub.load(module_url)
    return model(input)

def calculate_score(stud_answer,bot_answer):
    Data = pd.DataFrame(list(zip(stud_answer, bot_answer)), columns=['stud_answer', 'bot_answer'])
    ans = []

    for i in range(len(Data)):
        messages = [Data['stud_answer'][i], Data['bot_answer'][i]]               
        message_embeddings = embed(messages)                          
        a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings)) 
        cos_sim = dot(a[0], a[1])/(norm(a[0])*norm(a[1]))             
        ans.append(cos_sim)                

    Ans = pd.DataFrame(ans, columns = ['Similarity_Score'])
    Data = Data.join(Ans) 
    Data['Similarity_Score'] = Data['Similarity_Score'] + 1      
    Data['Similarity_Score'] = Data['Similarity_Score']/Data['Similarity_Score'].abs().max()

    score = Data['Similarity_Score'].mean()
    print(score)
    return score


if __name__ == "__main__":
    stud_answer = ['1. Can you tell us about your experience in front-end development?', '    2. How do you approach problem-solving in a project?', '    3. What kind of experience do you have with working collaboratively in agile teams?', '    4. How comfortable are you with back-end development?', '    5. Can you tell us about any experience you have with emerging technologies?', '    6. How do you stay up to date with the latest developments in technology?', '    7. How do you handle deadlines when working on a project?', '    8. What kind of projects have you worked on that required you to collaborate closely with team members from different departments?', '    9. Can you tell us about any challenges you faced while working on a project and how you overcame them?', '    1. Your experience in front-end development is mentioned in your resume. Can you elaborate on the specific skills or technologies you have used in this area?', '    2. The problem-solving skills mentioned in your resume are relevant to many technical roles. How do you apply these skills to a project?']
    bot_answer = [' I am a back-end engineer by profession, so my primary focus is on end and back-end development, with a keen interest in emerging technologies. However, I have also gained some experience in front-end development through personal projects and collaborations with other developers. In particular, I have worked with popular front-end frameworks like React and Angular, and have developed responsive user interfaces for web applications using HTML, CSS, and JavaScript. While my expertise is in back-end development, I am always looking to improve my skills and stay up-to-date with the latest trends in front-end development.', " As an experienced web developer with a Bachelor's degree in Computer Science, I approach problem-solving in a project by first identifying the root cause of the issue or problem at hand. This helps me better understand the issue and develop a plan to address it. Once I have a plan, I work closely with my team members to execute it and ensure that we meet our goals within the given timeframe. Throughout the process, I remain detail-oriented and continuously monitor the progress of the project to identify any potential issues or roadblocks that may arise. By taking this approach, I am able to consistently deliver high-quality code within tight deadlines and make meaningful contributions to innovative projects.", ' 3. I have experience working collaboratively in agile teams, particularly in software development projects. In my previous role as a developer, I worked closely with project managers and other team members to ensure that we met our project goals and delivered high-quality code within tight deadlines. I am skilled at communicating effectively, breaking down complex problems into manageable tasks, and adapting to changing requirements.', ' Very comfortable.', " 5. I am interested in emerging technologies, but I don't have direct experience working with them. However, as an end and back-end developer, I am always keeping up to date with the latest developments in the field of computer science and web development. I often incorporate innovative ideas and solutions into my work, and I am constantly learning new skills and technologies to improve my abilities.", ' Staying up to date with the latest developments in technology requires continuous learning, attending conferences and workshops, and reading relevant articles online or offline.', "   I make sure to prioritize tasks based on their importance and urgency. If there are any conflicts, I communicate with my team members to find a solution that works for everyone. I also regularly check in with stakeholders to ensure we're on track and discuss potential roadblocks.", " I haven't personally worked on any specific projects that required me to collaborate closely with team members from different departments. However, I am familiar with Agile methodology and have experience in collaborating with team members from diverse backgrounds in my previous roles.", '   While working on a project, I encountered some difficulties related to back-end development. Specifically, I had trouble integrating two APIs that were not compatible with each other. After many attempts at finding a solution, I decided to use a middleware tool that allowed me to translate the data from one API to another. This approach allowed me to successfully integrate the APIs and complete my project on time.', ' I have experience in using HTML, CSS, JavaScript, React, and Angular to create interactive web applications.', '     When working on a project, I use my problem-solving skills to analyze complex issues, break them down into smaller parts, and come up with creative solutions that align with the project requirements. I also leverage my technical expertise to identify potential roadblocks and develop workarounds that keep the project on track. By collaborating effectively with team members and communicating clearly, I am able to ensure that all stakeholders are informed of progress and any issues that arise. Ultimately, my focus is on delivering high-quality code on time, while continuously learning and improving my skills.']

    calculate_score(stud_answer,bot_answer)