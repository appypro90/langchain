from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template("tell me a short joke about {topic}")
prompt_value = prompt.invoke({"topic": "ice cream"})

prompt_value

"""Chat models uses prompt in the following signature: 
ChatPromptValue(messages=[HumanMessage(content='tell me a short joke about ice cream')])"""
print(prompt_value.to_messages())
print(prompt_value.to_string())

"""The PromptValue is then passed to model. In this case our model is a LLM, meaning it will output a string"""
from langchain_openai.llms import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")
response = llm.invoke(prompt_value)
print(response)