# Hackathon-Pubquiz

In this repository you can find starter code and data for the Pubquiz Bot!

Please create a virtual environment and install the requirements.txt:

```
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```

Make sure to create a .env file with 

```
AZURE_OPENAI_KEY=
AZURE_OPENAI_ENDPOINT=https://openai-workshop-genai.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt4o
AZURE_OPENAI_REASONING=o1-mini
AZURE_OPENAI_EMBEDDINGS=embeddings-large

```

which will be provided for you.

# Data

PubSql is a sql database which might be relevant for the Quiz

PubImages are images which might be relevant for the Quiz

PubTexts are texts which might be relevant for the Quiz

# Questions

examples_questions.txt are some possible questions for the quiz
