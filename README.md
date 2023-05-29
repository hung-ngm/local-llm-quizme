# Local LLM Quiz Me

## Generate a set of quizzes based on sample pdf file

## Set up

Create the `.env` file with the following configuration:

```
OPENAI_API_KEY=YOUR_API_KEY
PERSIST_DIRECTORY=db
```

## Run the program

- `python ingest.py`: To ingest the pdf file. You should change the file name in `ingest.py`
- `python script.py`: Run the program to generate a quiz
