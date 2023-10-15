# Image Classification Simple APP

This simple python app contains one predict api where the input is an image file and opencv model name used to predict.

Note: The ipynb file is me testing sample code before using it in main.py

## Running the app
```pip install -r requirments.txt
```

```uvicorn main:app --host 0.0.0.0 --reload```

you can try out the api by accessing
```http://0.0.0.0:8000/docs```
