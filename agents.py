from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI

from tools import GoogleShoppingTool
from tools import google_search

def create_agent():

    tools = [GoogleShoppingTool, google_search]

    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0).bind(functions=functions)

    prompt = ChatPromptTemplate.from_messages([("system", "You are fashion stylist assistant and will keep the conversation strictly related to fashion.If asked about weather you can search for weather data and suggest suitable clothes."),
                                               MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")])

    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

    chain = RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
                                      ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=chain, tools=tools, memory=memory, verbose=True)

    return agent_executor