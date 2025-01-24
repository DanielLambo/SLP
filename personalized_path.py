import streamlit as st
from dotenv import load_dotenv
import os

st.title("Personalized learning full-path recommendation model based on LSTM neural networks")
st.subheader("By: Daniel Lambo")
st.subheader("Mentor: Dr Yujian Fu")
st.divider()
st.header("Abstract:")
abstract  = '''This research investigates the potential of leveraging Artificial Intelligence (AI) to optimize student learning pathways. 
Specifically, we focus on employing Long Short-Term Memory (LSTM) networks within a machine learning framework to analyze and cluster learner data. 
By identifying meaningful trends and patterns in student learning behaviors, such as interaction patterns, engagement levels, and performance metrics, 
we aim to gain a deeper understanding of individual learning styles and preferences. 
This knowledge can then be used to personalize learning experiences, adapt instructional strategies, and ultimately improve learning outcomes. 
Our findings will contribute to the development of more effective and efficient learning environments that cater to the unique needs and abilities of each student.
'''
st.subheader(abstract)
st.divider()
st.header("Introduction: ")
intro = '''The rapid advancement of information and communication technologies has led to an abundance of learning resources, 
from web pages to multimedia content. Online learners generate vast amounts of data, 
offering valuable insights into learning patterns. Adaptive learning technologies (ALTs), 
powered by AI and machine learning, analyze user habits to provide personalized assistance, 
enhancing resource acquisition and learning outcomes. Academic libraries are evolving by integrating ALTs 
to better serve diverse user needs. Traditional library services often struggle to meet individual demands
 efficiently, but ALTs address these challenges by learning from user interactions and offering tailored
  recommendations. For instance, a student researching a specific topic can receive curated suggestions
   for materials that match their focus area, saving time and enhancing resource discovery.


'''
st.write(intro)

st.divider()
st.header("Assumptions: ")
Assumptions = '''based on the context of this paper , our general assumptions for a learner will be
1. Age Distribution and Learning Goals: Learners pursuing advanced topics like Deep Learning and Artificial Intelligence tend to be older, while those interested in foundational subjects such as Python Programming and Data Science are generally younger.

2. Skill Level and Learning Outcomes: There appears to be a positive correlation between higher skill levels and better learning outcomes, suggesting that more advanced learners achieve greater success in their educational pursuits.

3. Learning Path Complexity: Learners with higher skill levels often engage in more complex and longer learning paths, indicating a preference for in-depth study and comprehensive understanding of the subject matter.
'''
st.write("")




citation_one = '''Jayashri Bagade, Poonam Chaudhari, Poonam Girish Fegade, Ranjit M.
Gawande, Prachi P. Vast, Dipika R. Birari (2024). Adaptive Learning Technologies for Personalized
Research Assistance in Libraries. Library Progress International, 44(1), 242-258.

"7 Things You Should Know About Adaptive Learning." 
EDUCAUSE, 11 Jan. 2017, 
https://library.educause.edu/resources/2017/1/7-things-you-should-know-about-adaptive-learning. 
'''
st.divider(
)
st.write(Assumptions)
st.subheader("Works Cited:")
st.caption(citation_one)










































