# Automated Essay Scoring System

## Overview
This project aims to develop an automated essay scoring system using machine learning techniques and natural language processing (NLP) models. The system is designed to evaluate essays based on various criteria such as grammar, coherence, creativity, and similarity to reference texts. It utilizes a combination of feature engineering, deep learning models, and external libraries to achieve its goals.

## Motivation
The motivation behind this project stems from the significant time and effort educators invest in manually grading essays. Traditional grading methods are labor-intensive, subjective, and prone to inconsistencies. Automated essay scoring offers a solution to these challenges by leveraging advanced technologies to evaluate essays efficiently and effectively. By automating the grading process, educators can focus more on providing personalized feedback and supporting students' learning journey.

## Requirements and Constraints

### Requirements
- Python 3.9.2 
- Libraries and packages listed in _'requirements.txt'_
- Access to Kaggle Essay Scoring Dataset
- Pre-trained models (Word2Vec, BERT)
- Docker for containerization (optional)
  
### Constraints
- Limited computational resources
- Data privacy and security concerns
- Model fairness and bias mitigation
- Scalability for handling large volumes of essays


## Methodology

### Problem Statement
The primary objective is to develop a system capable of accurately grading essays based on various criteria such as content understanding, grammar, semantics, and overall quality. Additionally, the system should address challenges related to bias and fairness across different demographic groups and sensitivity to essay prompts.

### Data
The project uses a dataset of essays sourced from Kaggle, consisting of 12,000 essays across different topics and writing prompts. The dataset is preprocessed to correct spelling and grammar errors using the PySpellChecker library.

### Techniques
+ Word Embeddings: Essays are represented as word vectors using Word2Vec to capture semantic relationships between words.
+ Feature Engineering: Various features such as word count, grammar corrections, sentence count, and part-of-speech tags are extracted from the essays to provide additional context for the scoring model.
+ Deep Learning: A Multilayer Perceptron (MLP) model is used for scoring essays based on the extracted features and word embeddings.
+ Evaluation: The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and Cohen's Kappa score.

### Experimentation and Validation
The model is trained on a combination of training, validation, and test sets. Hyperparameters are tuned to optimize performance, and the model's accuracy is validated using cross-validation and test datasets. Additionally, the system incorporates feedback mechanisms to provide meaningful insights and feedback to users.

### Sub-Microservices
1. **Text Parsing**
   The Text Parsing microservice will be responsible for breaking down essays into manageable components for analysis. It will preprocess the text, tokenize sentences and words, and extract relevant features for further analysis.
2. **Grammar Checking**
   The Grammar Checking microservice will identify and correct grammatical errors in the essays. It will leverage language models and grammar rules to detect spelling mistakes, punctuation errors, and syntactic issues.
3. **Semantic Analysis**
   The Semantic Analysis microservice will focus on understanding the meaning and context of the essays. It will employ semantic similarity measures, topic modeling techniques, and sentiment analysis to assess the coherence, relevance, and clarity of the content.
4. **Scoring and Feedback Generation**
   The Scoring and Feedback Generation microservice will assign scores to the essays based on predefined criteria and generate constructive feedback for the students. It will consider various factors such as content quality, organization, creativity, and adherence to the prompt while providing feedback.

## Implementation
#### The implementation involves several steps:
1. Data preprocessing: Correcting spelling and grammar errors, extracting features, and generating word embeddings.
2. Model training: Training the MLP model on the preprocessed data.
3. Evaluation: Assessing the model's performance using appropriate metrics.
4. Deployment: Building a FastAPI-based web service and containerizing it with Docker for easy deployment and scalability.

### How to Use
#### Installation Steps
1. **Clone the Repository**: Clone the project repository to your local machine using Git.

   ```
    git clone <repository_url>
    ```
2. **Install Docker**: Install Docker on your system by following the instructions provided on the official Docker website ([https://docs.docker.com/get-docker/](url)).
3. **Build the Docker Image**: Navigate to the project directory and use the provided Dockerfile to build the Docker image.
   
   ```
   cd <project_directory>
   docker build -t essay-scoring-system .
   ```
4. **Run the Docker Container**: Once the Docker image is built, run the Docker container and specify the port to expose.

   ```
   docker run -d -p 8000:8000 essay-scoring-system
   ```
5. **Access the FastAPI Interface**: You can now access the FastAPI interface through a web browser by navigating to http://localhost:8000 or make HTTP requests programmatically to the API endpoint.
6. **Submit Essays**: Submit essays to the API endpoint and receive scoring results and feedback.

#### Input Method
To submit an essay for scoring, send a POST request to the API endpoint with the following JSON payload:

```
{
  "essay_content": "<your_essay_text_here>"
}
```
Replace '<your_essay_text_here>' with the content of the essay you want to score.

## Appendix
- Model Details: Detailed information about the MLP architecture, hyperparameters, and performance metrics.
- API Documentation: Documentation for the FastAPI endpoints and usage instructions.
- Docker Configuration: Instructions for building and running the Docker container.
- References: Citations and acknowledgments for external libraries, datasets, and resources used in the project.


















