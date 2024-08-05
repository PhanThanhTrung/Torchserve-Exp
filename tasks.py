from celery_worker import app as celery_worker
from typing import List

@celery_worker.task()
def text_embedding(input_text: str):
    pass

@celery_worker.task()
def rerank(query_text: str, passage_texts: List[str]):
    pass

@celery_worker.task()
def compute_score(query_text: str, passage: str):
    pass

