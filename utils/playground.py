from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser
input_prompt = "What is COVID-19 ?"
llm_response = chain.invoke({"text": input_prompt})
print(llm_response)