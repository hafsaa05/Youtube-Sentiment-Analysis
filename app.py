from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from predict import predict
import os

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class CommentRequest(BaseModel):
    comment: str

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.post("/analyze")
def analyze(req: CommentRequest):
    result = predict(req.comment)
    return result