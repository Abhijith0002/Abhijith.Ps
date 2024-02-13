# Automated Essay Scoring System

## Overview
This project aims to develop an automated essay scoring system using machine learning techniques and natural language processing (NLP) models. The system is designed to evaluate essays based on various criteria such as grammar, coherence, creativity, and similarity to reference texts. It utilizes a combination of feature engineering, deep learning models, and external libraries to achieve its goals.

## Motivation
Automated essay scoring systems offer several advantages, including efficiency, consistency, and scalability. By automating the essay grading process, educational institutions and testing organizations can save time and resources while ensuring fair and unbiased evaluations. Moreover, such systems can provide immediate feedback to students, helping them improve their writing skills.

## Requirements and Constraints
### Dependencies
- Python 3.9.2
- Docker
### Python Libraries
- FastAPI
- TensorFlow
- Gensim
- Spacy
- NLTK
- PySpellChecker
- Scikit-learn
- Matplotlib
- Pandas
- NumPy

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


















