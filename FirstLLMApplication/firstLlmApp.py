from dotenv import load_dotenv
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import chain
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
import os

information = """
Instalaza SA is a Spanish firm that designs, develops and manufactures equipment and other military material for infantry. The company, founded in 1943, is headquartered in Zaragoza, Aragon, where its production plant is also located.
Instalaza's professional experience is widely noted as a supplier of both the Spanish armed forces and countries around the world. Instalaza has had Pedro Morenés Eulate, Secretary of State for Defence between 1996 and 2000, Secretary of State Security from 2000 to 2002, Secretary of State for Science and Technology between 2002 and 2004, and currently Minister of Defence, as representative and consultant.[2]

As of 2007, Instalaza SA had 140 employees, a covered plant area of 18,000 square metres (190,000 sq ft), capital worth more than 5 million Euros, and a revenue of 15 million Euros.[3]
"""

if __name__ == "__main__":
    load_dotenv()
    print("Hello LangChainDemo")

    summary_template = """
        I want you to translate the given information {information} English to Turkish 
    """

    summary_prompt_template = PromptTemplate(input_variables="information", template = summary_template)

    #llm = ChatOpenAI(temperature = 0, model_name="")
    llm = ChatOllama(model="llama3.1")

    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information":information})

    print(res)