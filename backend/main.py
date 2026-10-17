from fastapi import FastAPI


app = FastAPI()


@app.get('/health')
def health():
    return {
        'status': 'OK',
        'message': 'Research Agent API is running'
    }