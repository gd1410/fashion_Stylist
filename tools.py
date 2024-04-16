import os
import requests
from PIL import Image

from langchain_community.utilities import SearchApiAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent

from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)

# Schema definition for Google Shopping API tool
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")

# Custom Google Shopping API Tool
@tool(args_schema=SearchInput)
def GoogleShoppingTool(query:str):
    """use the tool when you need to search for products on google shoppings api"""
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google_shopping",
        "q": query,
        "location": "India",
        "gl":"in",
        "num":"3",
        "api_key":os.getenv("SEARCHAPI_API_KEY")
    }

    response = requests.get(url, params = params)
    if response.status_code == 200:
        data = response.json()
    else:
        raise Exception(f"API Request failed with status cod:{response.status_code}")
    
    thumbnail_img = [d['thumbnail'] for d in data["shopping_results"][:2] if 'thumbnail' in d]
    return (data['shopping_results'][:2],thumbnail_img)

# Google Search Tool
@tool
def google_search(query: str):
    """Performs a Google search using the provided query string.
    Choose this tool when you need to find weather data or data about the place such as its current fashion trend
    When asked about weather data use www.accuweather.com to get the temperature data """
    return SerpAPIWrapper().run(query)




# Generate your own image - tool
@tool
def fashion_image_generator(query: str):
    """Call this when the user wants to generate his custom fashion image"""
    res = DallEAPIWrapper(model="dall-e-3").run(f"You generate a single image of specific clothes with the mentioned color combination, text and pattern as requested. Strictly Do not generate any obscene or vulgar image or text. Strictly Do not generate any discriminative image.")

    answer_to_agent = (f"Use this format- Here is your custom fashion generated - did you like it?"
                       f"url= {res}")
    return answer_to_agent