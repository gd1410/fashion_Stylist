from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_openai.chat_models import ChatOpenAI

from tools import GoogleShoppingTool
from tools import google_search, fashion_image_generator

def create_agent():

    tools = [GoogleShoppingTool, google_search, fashion_image_generator]

    functions = [convert_to_openai_function(f) for f in tools]
    model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0).bind(functions=functions)
    sys_prompt= """"You are fashion stylist assistant named STYLO and you will keep the conversation strictly related to fashion.
    If asked to find any clothes such as tshirts , shirts or sarees, kurtas or other wear ask more information such as what pattern, what price range, what occasion and use this information to find the relevant clothes on google shopping.
    If asked about weather you can search for weather data and suggest suitable clothes.
    If you are asked to generate or create custom clothes ask for more details such as what should be printed, what color, any text needs to be added etc."
"""

    prompt = ChatPromptTemplate.from_messages([("system", sys_prompt),
                                               MessagesPlaceholder(variable_name="chat_history"), ("user", "{input}"),
                                               MessagesPlaceholder(variable_name="agent_scratchpad")])

    memory = ConversationBufferWindowMemory(return_messages=True, memory_key="chat_history", k=5)

    chain = RunnablePassthrough.assign(agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
                                      ) | prompt | model | OpenAIFunctionsAgentOutputParser()

    agent_executor = AgentExecutor(agent=chain, tools=tools, memory=memory, verbose=True)

    return agent_executor