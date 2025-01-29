import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
import os

st.title(":red[Personalized learning full-path recommendation model based on LSTM neural networks]")
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
st.write(Assumptions)

st.divider()
st.header("Implementation")
st.subheader(":red[Data Anaylsis]")
st.write("We will begin with explaining the input data we are analyzing in this project.")
data_explanation ='''
"This study will utilize three distinct datasets to train and evaluate a personalized learning path recommendation system within a computer and software engineering institution. 

1. :red[**Learner Dataset:**] This dataset will encompass information about each learner, including:
    * `learner_id`: Unique identifier for each learner.
    * `age`: Age of the learner.
    * `learning_goal`: The specific learning objective of the learner (e.g., "Master Data Structures," "Prepare for a software engineering interview").
    * `current_skill_level`: Quantified assessment of the learner's current proficiency in the relevant domain (e.g., beginner, intermediate, advanced).

2. :red[**Learning Path Dataset:**] This dataset will describe historical learning paths undertaken by learners, including:
    * `learner_id`: Linking the learning path to the corresponding learner.
    * `sequence_of_learning`: An ordered list of learning resources (courses, articles, exercises) followed by the learner.
    * `learning_outcome`: A numerical score (0-1) representing the learner's achieved proficiency level after completing the learning path.

3. :red[**Knowledge Point Dataset:**] This dataset will characterize the knowledge points covered within each course, including:
    * `knowledge_point_id`: Unique identifier for each knowledge point.
    * `topic`: The specific subject area of the knowledge point (e.g., "Object-Oriented Programming," "Algorithms").
    * `difficulty`: A measure of the complexity of the knowledge point (e.g., easy, medium, hard).

These datasets will serve as crucial inputs for the machine learning model to accurately predict and recommend personalized learning paths tailored to individual learner needs and goals within the context of a computer and software engineering educational setting.


By analyzing the GitHub data, we can generate a bar chart visualizing the relationship between learner courses and their current skill levels.
'''
st.write(data_explanation)

learners = [
    {'learner_id': 1, 'age': 25, 'learning_goal': 'Machine Learning', 'skill_level': 'Intermediate'},
    {'learner_id': 2, 'age': 22, 'learning_goal': 'Data Science', 'skill_level': 'Beginner'},
    {'learner_id': 3, 'age': 30, 'learning_goal': 'Deep Learning', 'skill_level': 'Advanced'},
    {'learner_id': 4, 'age': 27, 'learning_goal': 'Artificial Intelligence', 'skill_level': 'Intermediate'},
    {'learner_id': 5, 'age': 20, 'learning_goal': 'Data Science', 'skill_level': 'Beginner'},
    {'learner_id': 6, 'age': 35, 'learning_goal': 'Deep Learning', 'skill_level': 'Advanced'},
    {'learner_id': 7, 'age': 23, 'learning_goal': 'Python Programming', 'skill_level': 'Beginner'},
    {'learner_id': 8, 'age': 28, 'learning_goal': 'Machine Learning', 'skill_level': 'Intermediate'},
    {'learner_id': 9, 'age': 26, 'learning_goal': 'Statistics', 'skill_level': 'Beginner'},
    {'learner_id': 10, 'age': 29, 'learning_goal': 'Artificial Intelligence', 'skill_level': 'Advanced'},
    {'learner_id': 11, 'age': 21, 'learning_goal': 'Data Science', 'skill_level': 'Beginner'},
    {'learner_id': 12, 'age': 32, 'learning_goal': 'Machine Learning', 'skill_level': 'Advanced'},
    {'learner_id': 13, 'age': 24, 'learning_goal': 'Deep Learning', 'skill_level': 'Intermediate'},
    {'learner_id': 14, 'age': 38, 'learning_goal': 'Artificial Intelligence', 'skill_level': 'Advanced'},
    {'learner_id': 15, 'age': 19, 'learning_goal': 'Python Programming', 'skill_level': 'Beginner'},
    {'learner_id': 16, 'age': 27, 'learning_goal': 'Statistics', 'skill_level': 'Intermediate'},
    {'learner_id': 17, 'age': 31, 'learning_goal': 'Machine Learning', 'skill_level': 'Advanced'},
    {'learner_id': 18, 'age': 25, 'learning_goal': 'Data Science', 'skill_level': 'Intermediate'},
    {'learner_id': 19, 'age': 29, 'learning_goal': 'Deep Learning', 'skill_level': 'Beginner'},
    {'learner_id': 20, 'age': 33, 'learning_goal': 'Artificial Intelligence', 'skill_level': 'Intermediate'},
    {'learner_id': 21, 'age': 22, 'learning_goal': 'Python Programming', 'skill_level': 'Intermediate'},
    {'learner_id': 22, 'age': 30, 'learning_goal': 'Statistics', 'skill_level': 'Advanced'},
    {'learner_id': 23, 'age': 26, 'learning_goal': 'Machine Learning', 'skill_level': 'Beginner'},
    {'learner_id': 24, 'age': 28, 'learning_goal': 'Data Science', 'skill_level': 'Advanced'},
    {'learner_id': 25, 'age': 35, 'learning_goal': 'Deep Learning', 'skill_level': 'Intermediate'},
    {'learner_id': 26, 'age': 21, 'learning_goal': 'Artificial Intelligence', 'skill_level': 'Beginner'},
    {'learner_id': 27, 'age': 24, 'learning_goal': 'Python Programming', 'skill_level': 'Advanced'},
    {'learner_id': 28, 'age': 32, 'learning_goal': 'Statistics', 'skill_level': 'Intermediate'},
    {'learner_id': 29, 'age': 27, 'learning_goal': 'Machine Learning', 'skill_level': 'Beginner'},
    {'learner_id': 30, 'age': 30, 'learning_goal': 'Data Science', 'skill_level': 'Intermediate'}
]
learning_paths = [
    {'learner_id': 1, 'sequence': ['ML01', 'ML02', 'ML03'], 'learning_outcome': 0.8},
    {'learner_id': 2, 'sequence': ['DS01', 'DS02'], 'learning_outcome': 0.6},
    {'learner_id': 3, 'sequence': ['DL01', 'DL02'], 'learning_outcome': 0.9},
    {'learner_id': 4, 'sequence': ['AI01', 'AI02', 'ML01'], 'learning_outcome': 0.75},
    {'learner_id': 5, 'sequence': ['DS01', 'ST01'], 'learning_outcome': 0.5},
    {'learner_id': 6, 'sequence': ['DL01', 'ML03', 'DL02'], 'learning_outcome': 0.95},
    {'learner_id': 7, 'sequence': ['ST01', 'DS01'], 'learning_outcome': 0.4},
    {'learner_id': 8, 'sequence': ['ML01', 'ML02'], 'learning_outcome': 0.7},
    {'learner_id': 9, 'sequence': ['ST01', 'ML01'], 'learning_outcome': 0.65},
    {'learner_id': 10, 'sequence': ['AI01', 'DL01'], 'learning_outcome': 0.85},
    {'learner_id': 11, 'sequence': ['DS01', 'DS02', 'DS03'], 'learning_outcome': 0.7},
    {'learner_id': 12, 'sequence': ['ML01', 'ML02', 'ML03', 'ML04'], 'learning_outcome': 0.85},
    {'learner_id': 13, 'sequence': ['DL01', 'DL02', 'DL03'], 'learning_outcome': 0.92},
    {'learner_id': 14, 'sequence': ['AI01', 'AI02', 'AI03'], 'learning_outcome': 0.88},
    {'learner_id': 15, 'sequence': ['ST01', 'ST02', 'ST03'], 'learning_outcome': 0.6},
    {'learner_id': 16, 'sequence': ['ML01', 'DS01', 'ML02'], 'learning_outcome': 0.78},
    {'learner_id': 17, 'sequence': ['DL01', 'ML03', 'DL02'], 'learning_outcome': 0.9},
    {'learner_id': 18, 'sequence': ['AI01', 'ML01', 'AI02'], 'learning_outcome': 0.82},
    {'learner_id': 19, 'sequence': ['DS01', 'ST01', 'DS02'], 'learning_outcome': 0.65},
    {'learner_id': 20, 'sequence': ['ML01', 'ML02', 'ML03', 'ML04'], 'learning_outcome': 0.9},
    {'learner_id': 21, 'sequence': ['DL01', 'DL02', 'ML01'], 'learning_outcome': 0.88},
    {'learner_id': 22, 'sequence': ['AI01', 'AI02', 'AI03', 'ML01'], 'learning_outcome': 0.95},
    {'learner_id': 23, 'sequence': ['ST01', 'ST02', 'DS01'], 'learning_outcome': 0.55},
    {'learner_id': 24, 'sequence': ['ML01', 'ML02', 'DS01'], 'learning_outcome': 0.72},
    {'learner_id': 25, 'sequence': ['DL01', 'DL02', 'DL03'], 'learning_outcome': 0.98},
    {'learner_id': 26, 'sequence': ['AI01', 'ML01', 'AI02'], 'learning_outcome': 0.8},
    {'learner_id': 27, 'sequence': ['ST01', 'ST02', 'ST03'], 'learning_outcome': 0.7},
    {'learner_id': 28, 'sequence': ['ML01', 'DS01', 'ML02'], 'learning_outcome': 0.85},
    {'learner_id': 29, 'sequence': ['DL01', 'ML03', 'DL02'], 'learning_outcome': 0.92},
    {'learner_id': 30, 'sequence': ['AI01', 'AI02', 'ML01'], 'learning_outcome': 0.78}
]
knowledge_points = [
    {'kp_id': 'ML01', 'topic': 'Python Basics', 'difficulty': 'Easy'},
    {'kp_id': 'ML02', 'topic': 'Linear Regression', 'difficulty': 'Intermediate'},
    {'kp_id': 'ML03', 'topic': 'Neural Networks', 'difficulty': 'Advanced'},
    {'kp_id': 'DS01', 'topic': 'Data Cleaning', 'difficulty': 'Easy'},
    {'kp_id': 'DS02', 'topic': 'Exploratory Data Analysis', 'difficulty': 'Intermediate'},
    {'kp_id': 'DL01', 'topic': 'Convolutional Neural Networks', 'difficulty': 'Advanced'},
    {'kp_id': 'DL02', 'topic': 'Recurrent Neural Networks', 'difficulty': 'Advanced'},
    {'kp_id': 'AI01', 'topic': 'Introduction to AI', 'difficulty': 'Easy'},
    {'kp_id': 'AI02', 'topic': 'AI Ethics', 'difficulty': 'Intermediate'},
    {'kp_id': 'ST01', 'topic': 'Probability Basics', 'difficulty': 'Easy'},
    {'kp_id': 'ML04', 'topic': 'Logistic Regression', 'difficulty': 'Intermediate'},
    {'kp_id': 'ML05', 'topic': 'Decision Trees', 'difficulty': 'Intermediate'},
    {'kp_id': 'DS03', 'topic': 'Data Visualization', 'difficulty': 'Intermediate'},
    {'kp_id': 'DS04', 'topic': 'Feature Engineering', 'difficulty': 'Advanced'},
    {'kp_id': 'DL03', 'topic': 'Generative Adversarial Networks (GANs)', 'difficulty': 'Advanced'},
    {'kp_id': 'DL04', 'topic': 'Transformer Networks', 'difficulty': 'Advanced'},
    {'kp_id': 'AI03', 'topic': 'Natural Language Processing (NLP)', 'difficulty': 'Advanced'},
    {'kp_id': 'AI04', 'topic': 'Computer Vision', 'difficulty': 'Advanced'},
    {'kp_id': 'ST02', 'topic': 'Statistical Inference', 'difficulty': 'Intermediate'},
    {'kp_id': 'ST03', 'topic': 'Hypothesis Testing', 'difficulty': 'Intermediate'},
    {'kp_id': 'ML06', 'topic': 'Support Vector Machines (SVM)', 'difficulty': 'Advanced'},
    {'kp_id': 'DS05', 'topic': 'Time Series Analysis', 'difficulty': 'Advanced'},
    {'kp_id': 'DL05', 'topic': 'Reinforcement Learning', 'difficulty': 'Advanced'},
    {'kp_id': 'AI05', 'topic': 'Robotics', 'difficulty': 'Advanced'},
    {'kp_id': 'ST04', 'topic': 'Bayesian Statistics', 'difficulty': 'Advanced'},
    {'kp_id': 'ML07', 'topic': 'Ensemble Methods', 'difficulty': 'Advanced'},
    {'kp_id': 'DS06', 'topic': 'Big Data Analytics', 'difficulty': 'Advanced'},
    {'kp_id': 'DL06', 'topic': 'Autoencoders', 'difficulty': 'Advanced'},
    {'kp_id': 'AI06', 'topic': 'Explainable AI (XAI)', 'difficulty': 'Advanced'},
    {'kp_id': 'ST05', 'topic': 'Non-parametric Statistics', 'difficulty': 'Advanced'}
]

learners_df = pd.DataFrame(learners)
fig = px.bar(
    learners_df,
    x="learning_goal",
    color="skill_level",
    title="Learning Goals by Skill Level",
    labels={"learning_goal": "Learning Goal", "count": "Number of Learners"}
)
st.plotly_chart(fig)


st.write("Based on the second dataset, we can visualize the relationship between learners course path length(The number of total courses a learner takes in a sequence) and the learner's final outcome(the degree of how much the learner assimilated): ")

learning_paths_df = pd.DataFrame(learning_paths)

learning_paths_df['sequence_length'] = learning_paths_df['sequence'].apply(len)

outcome_plot = px.scatter(
    learning_paths_df,
    x='sequence_length',
    y='learning_outcome',
    title='Learning Outcome vs. Sequence Length',
    labels={'sequence_length': 'Number of Courses', 'learning_outcome': 'Learning Outcome'},
   trendline='ols'  # Add a trendline to show correlation
)
knowledge_points_df = pd.DataFrame(knowledge_points)
# Display the plot in Streamlit
st.plotly_chart(outcome_plot)
st.write("It is not a very interesting graph, but it shows the correlation!")
st.write("\n")
st.write("We also have a knowledge point dataset head that looks like this ")
st.write(knowledge_points_df.head())

st.divider()
st.header(":blue[Data Preprocessing]")
st.write("Now that we have analyzed all the data that will be utlized for this project, we will proceed to preprocess our existing data.")
Preprocessing_phase = '''
Data cleaning is performed on the imported pandas DataFrames. Rows or columns containing null or empty values are removed to ensure data integrity. 

Feature engineering is conducted. String-based features in the learner data (e.g., "Machine Learning") are converted into numerical representations for compatibility with most machine learning algorithms. The `goal_encoder.fit_transform()` method is employed to map unique categories within the 'learning_goals' column to numerical labels. Knowledge points are encoded using `LabelEncoder` to assign unique numerical identifiers to each distinct knowledge point.

The learner, learning path, and knowledge point datasets are merged into a single DataFrame (`merged_df`) to establish comprehensive learner profiles. The `merged_df` integrates learner attributes (age, learning_goal, etc.) with their learning paths and the corresponding encoded knowledge points. 

The merged data is transformed into sequences of learned knowledge points. Each sequence represents a partial learning path, and the corresponding output is the next expected knowledge point in the sequence. To accommodate deep learning models, sequences are padded to ensure uniform length.

The dataset is divided into training and test sets for model evaluation. The data is then converted into TensorFlow Datasets and batched for efficient processing during the training phase. 

A portion of the processed data is printed to visually inspect the sequences and labels, ensuring their accuracy and readiness for model training.

This preprocessed data enables the model to learn patterns in learner behavior and predict the most suitable next step in their learning journey based on their prior learning history and individual characteristics.

'''
st.write(Preprocessing_phase)

st.divider()
st.header(":blue[Data Clustering]")

clustering_text = '''
Now that the data has been fully prepared and processed, the next step is clustering, which involves segmenting learners into meaningful groups based on similar learning characteristics. 
The primary advantage of clustering is that it helps identify common learning patterns, allowing for the development of targeted learning strategies.  

To achieve this, a clustering class will be created with key variables, including the learners' dataframe, learning paths, preprocessed data, clustering results, and feature columns. 
The first method in this class will focus on preprocessing and feature engineering for clustering mixed data types.  

We begin by identifying numerical features, such as age and learning outcomes, while categorical features include skill level, learning goals, learning style, and level. 
A preprocessing pipeline will be implemented to handle missing values by replacing them with the string "missing." Additionally, one-hot encoding will be applied to convert categorical variables into binary columns, ensuring compatibility for clustering algorithms.
 Finally, the processed data will be stored, and feature names will be prepared for further use.


The precise clustering Algorithm we will be using is the **K-Means CLustering Algorithm**.
This is an unsupervised machine learning algorithm used to group data into **K** distinct clusters.

It works simply by initializing K centorids randomly, assigning each data point to the nearest centroid,
recalculating centroids by computing means of all points in each cluster.
and repeat. The algorithm aims to minimize intra-cluster variance, ensuring that data points within the same cluster are as similar as possible.
, K-Means clustering is used to segment learners into groups based on their characteristics (e.g., age, skill level, learning goals). This helps:

Identify distinct learner profiles, enabling personalized recommendations.
Analyze learning patterns, improving curriculum design.
Optimize targeted learning strategies, ensuring better engagement.
Evaluates clustering quality using:
Silhouette Score (measures how well points fit within clusters).
Calinski-Harabasz Score (evaluates cluster dispersion).
Stores the results, including cluster labels and centroids, for further analysis.
The code snippet for the clustering algorithm is below
'''

st.write(clustering_text)


clustering_code = '''
    def perform_kmeans_clustering(self, n_clusters=3):
        """
        Perform K-Means clustering

        Parameters:
        - n_clusters: Number of clusters to form

        Returns:
        - Cluster labels and clustering metrics
        """
        if self.preprocessed_data is None:
            self.preprocess_features()

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )

        kmeans_labels = kmeans.fit_predict(self.preprocessed_data)

        # Clustering evaluation metrics
        silhouette = silhouette_score(
            self.preprocessed_data,
            kmeans_labels
        )

        calinski = calinski_harabasz_score(
            self.preprocessed_data,
            kmeans_labels
        )

        self.clustering_results['kmeans'] = {
            'labels': kmeans_labels,
            'silhouette_score': silhouette,
            'calinski_score': calinski,
            'centroids': kmeans.cluster_centers_
        }

        return kmeans_labels

'''

st.code(clustering_code,language = "python")
more_stuff = '''
    The cluser output would be visualized using Principle component analysis, the code for this can be seen in the GitHub, another method would be utilized to create cluster profiles, and one more to analyze cluster
    characteristics


'''

citation_one = '''Jayashri Bagade, Poonam Chaudhari, Poonam Girish Fegade, Ranjit M.
Gawande, Prachi P. Vast, Dipika R. Birari (2024). Adaptive Learning Technologies for Personalized
Research Assistance in Libraries. Library Progress International, 44(1), 242-258.

"7 Things You Should Know About Adaptive Learning." 
EDUCAUSE, 11 Jan. 2017, 
https://library.educause.edu/resources/2017/1/7-things-you-should-know-about-adaptive-learning. 
'''
st.divider()
st.subheader("Works Cited:")
st.caption(citation_one)