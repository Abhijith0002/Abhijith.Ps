# Automated Essay Scoring System

## Overview
This project aims to develop an automated essay scoring system using machine learning techniques and natural language processing (NLP) models. The system is designed to evaluate essays based on various criteria such as grammar, coherence, creativity, and similarity to reference texts. It utilizes a combination of feature engineering, deep learning models, and external libraries to achieve its goals.

## Motivation
The motivation behind this project stems from the significant time and effort educators invest in manually grading essays. Traditional grading methods are labor-intensive, subjective, and prone to inconsistencies. Automated essay scoring offers a solution to these challenges by leveraging advanced technologies to evaluate essays efficiently and effectively. By automating the grading process, educators can focus more on providing personalized feedback and supporting students' learning journey.

## Requirements and Constraints

### Requirements
- Python 3.9.2 
- Libraries and packages listed in '***requirements.txt***'
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
The project employs a combination of machine learning models, including Multi-Layer Perceptron (MLP), Word2Vec, and BERT, for essay evaluation. Model performance is evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and Cohen's Kappa score. Cross-validation techniques ensure robustness and generalization of models.

### Sub-Microservices
1. **Text Parsing**
   The Text Parsing microservice will be responsible for breaking down essays into manageable components for analysis. It will preprocess the text, tokenize sentences and words, and extract relevant features for further analysis.
2. **Grammar Checking**
   The Grammar Checking microservice will identify and correct grammatical errors in the essays. It will leverage language models and grammar rules to detect spelling mistakes, punctuation errors, and syntactic issues.
3. **Semantic Analysis**
   The Semantic Analysis microservice will focus on understanding the meaning and context of the essays. It will employ semantic similarity measures, topic modeling techniques, and sentiment analysis to assess the coherence, relevance, and clarity of the content.
4. **Scoring and Feedback Generation**
   The Scoring and Feedback Generation microservice will assign scores to the essays based on predefined criteria and generate constructive feedback for the students. It will consider various factors such as content quality, organization, creativity, and adherence to the prompt while providing feedback.

## Implementation Details

### Model Architecture
#### Multi-Layer Perceptron (MLP)
- The MLP model is utilized for regression tasks, predicting essay scores based on extracted features.
- The architecture consists of multiple fully connected layers with ReLU activation functions.
- Dropout regularization is applied to prevent overfitting.
- The output layer provides the final score prediction.

#### Word2Vec Embeddings
- Word2Vec is employed to generate word embeddings capturing semantic similarity and context in essays.
- The Word2Vec model is trained on the cleaned essay text to learn word representations.
- These embeddings are then used as features for essay evaluation.

#### BERT (Bidirectional Encoder Representations from Transformers)
- BERT is a powerful pre-trained language model used for semantic understanding and contextualized word embeddings.
- The BERT model is fine-tuned on essay data to capture intricate relationships and nuances in language.
- It helps improve the accuracy and effectiveness of essay scoring and feedback generation.

### Features
##### A comprehensive set of features is extracted from essays to facilitate evaluation:
- Word count
- Grammar corrections
- Sentence count
- Named Entity Recognition (NER) count
- Part-of-speech tags
- Semantic similarity
- Coherence score

### FastAPI Integration
A FastAPI server is implemented to expose the AES system as a RESTful API. Users can submit essays for scoring and receive feedback in real-time, making the system accessible and user-friendly.

### Dockerization
The project is containerized using Docker, allowing for easy deployment and scalability. Docker ensures consistency across different environments and simplifies the deployment process, making it suitable for production environments.

## Model Performance

### MLP Model
+ Achieves a Cohen's Kappa score of 0.97, indicating high agreement with human raters.
+ Demonstrates robustness and generalization across different essay prompts and topics.

### Word2Vec and BERT
+ Word2Vec and BERT embeddings significantly enhance the quality of essay evaluation.
+ They capture semantic similarity, contextual understanding, and nuanced language nuances, improving scoring accuracy.

## How to Use
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

## Future Enhancements

### Advanced NLP Techniques
+ Integration of advanced NLP techniques, such as attention mechanisms and transformer architectures, to further enhance semantic understanding and essay evaluation.
### Prompt-Specific Models
+ Fine-tuning models for specific essay prompts and topics to improve performance and accuracy.
### User Feedback Mechanism
+ Incorporating user feedback to continuously refine and enhance the AES system, ensuring relevance and effectiveness in real-world scenarios.

## Conclusion
Automated Essay Scoring represents a transformative advancement in educational technology, offering a scalable, efficient, and objective solution to essay grading. By leveraging machine learning and NLP techniques, this project demonstrates the feasibility of automating essay evaluation while ensuring fairness, accuracy, and reliability. Continued research and development in this field hold the potential to revolutionize the education system and enhance learning outcomes for students worldwide.

















