from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

class QueryExpander:
    def __init__(self, llm):
        self.llm = llm
        self.expansion_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query expansion expert. Given a user's question, generate 3 alternative ways to phrase the question 
            that might help retrieve more relevant information. Each alternative should:
            1. Maintain the original intent
            2. Use different terminology or phrasing
            3. Be more specific or broader when appropriate
            4. Consider different aspects of the question
            
            Return the alternatives as a numbered list."""),
            ("human", "{input}")
        ])
        
    def expand_query(self, query: str) -> List[str]:
        # Generate alternative queries
        response = self.llm.invoke(self.expansion_prompt.format_messages(input=query))
        
        # Parse the response to get individual queries
        alternatives = []
        for line in response.content.split('\n'):
            if line.strip() and line[0].isdigit():
                # Remove the number and any following punctuation
                alternative = line.split('.', 1)[1].strip()
                alternatives.append(alternative)
        
        # Include the original query
        return [query] + alternatives[:3]  # Return original + top 3 alternatives 