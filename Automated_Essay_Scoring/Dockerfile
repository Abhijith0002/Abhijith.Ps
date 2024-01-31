FROM python:3.8.2

ADD main.py .

ADD models.py .

ADD SavedModels /SavedModels 

ADD requirements.txt . 

RUN pip install --upgrade pip

# # Upgrade Gensim before installing other dependencies
# RUN pip install --no-cache-dir gensim --upgrade

RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords')"

CMD ["python", "main.py"]