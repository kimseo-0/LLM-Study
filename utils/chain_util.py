from dotenv import load_dotenv
load_dotenv()

import os
project_name = "prompt_basic"
os.environ["LANGSMITH_PROJECT"] = project_name

from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    temperature=0.1, # 창의력 정도
    model = "gpt-4.1-mini",
    verbose=True
)

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_basicTemplate_chain(template):
    prompt = PromptTemplate.from_template(template)
    prompt

    # 출력 파서 
    outputparser = StrOutputParser()

    chain = prompt | model | outputparser
    return chain