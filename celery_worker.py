import os
from celery import Celery
from dotenv import load_dotenv
load_dotenv()

# The app.task() decorators don’t create the tasks at the point when the task is defined, 
# instead it’ll defer the creation of the task to happen either when the task is used, or 
# after the application has been finalized.

class Config:
    enable_utc = True
    timezone = "Asia/Bangkok"

app = Celery('Text Utilities', broker=os.environ.get("BROKER_URL"))
