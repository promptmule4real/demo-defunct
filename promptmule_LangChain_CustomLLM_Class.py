"""
This Python class, named `promptmuleLLM`, serves as a wrapper around the LangChain's Language Learning Model (LLM) base class.
The class overloads the `_call` method to make an API request to a specified endpoint. The request is built using the 
given prompt and headers (which include API token, API key, and OpenAI key). After sending the request, the class parses 
the response to extract and return the content.

This class also overrides the `_identifying_params` method to return a dictionary of the identifying parameters. These 
parameters are the API token, API key, OpenAI key, and username, which are necessary for making the API requests.

When creating an instance of this class, replace the placeholders 'api_token', 'api_key', 'openai_key', and 'username' 
with your actual credentials. For example:

llm = promptmuleLLM(api_token='your_api_token', api_key='your_api_key', openai_key='your_openai_key', username='your_username')

To use this class, you can call it with a string, which it treats as a prompt. For example:

print(llm("Create a Psuedocode function for the chances of seeing a unicorn in Central Park. Assume the probability is the square root of -1."))

This will print the response from the API.

Ensure that the 'requests' library is installed in your Python environment. If it's not, you can install it by running 'pip install requests' in your terminal.

Please note, this is a basic implementation and doesn't include advanced features such as error handling, API limit management, token management, etc. For more robust applications, it's advisable to extend this class or add those features according to the application's specific needs.
"""

import requests
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

class PromptMuleLLM(LLM):
 """
    Custom Language Learning Model (LLM) that interfaces with a specific API endpoint to generate model responses.

    Attributes:
        headers (dict): The headers to include in the API requests. Includes authorization and API keys.
        api_url (str): The URL of the API endpoint to interface with.
        username (str): The username to be used for the API requests.
        """
    def __init__(self, api_token: str, api_key: str, openai_key: str, username: str):
        """
        Initialize the custom LLM with the given API keys and username.

        Args:
            api_token (str): The API token for authorization.
            api_key (str): The API key.
            openai_key (str): The OpenAI API key.
            username (str): The username to be used for the API requests.
            """
        self.headers = {           
            'Authorization': api_token,
            'x-api-key': api_key,
            'openai-key': openai_key,
            'Content-Type': 'application/json'
        }
        self.api_url = 'https://820czjhki0.execute-api.us-west-2.amazonaws.com/dev/prompt'
        self.username = username

    @property
    def _llm_type(self) -> str:
    """Return the type of the LLM (always "custom" for this custom LLM)."""
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
    """
        Make a POST request to the API endpoint with the given prompt.

        Args:
            prompt (str): The prompt to generate a response for.
            stop (Optional[List[str]]): A list of stop words. If provided, a ValueError is raised, as stop words are not supported.
            run_manager (Optional[CallbackManagerForLLMRun]): An optional run manager. This is currently unused.

        Returns:
            str: The response from the API endpoint.

        Raises:
            ValueError: If stop words are provided, as they are not supported.
            """
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        body = {
            "model": "gpt-4-0613",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": "100",
            "temperature": "0.99",
            "top_p": "1",
            "n": "1",
            "logprobs": "null",
            "stop": "null",
            "suffix": "null",
            "echo": "true",
            "presence_penalty": "0",
            "frequency_penalty": "0",
            "best_of": "1",
            "logit_bias": "null",
            "user": self.username,
            "api": "openai",
            "semantic": "0.99",
            "sem_num": "2"
        }
        
        response = requests.post(self.api_url, headers=self.headers, json=body)
        response.raise_for_status() # Check if request was successful
        
        data = response.json()
        return data['message']['content']

    @property
    def _identifying_params(self) -> Dict[str, Any]:
    """
        Get the identifying parameters.

        Returns:
            Dict[str, Any]: A dictionary containing the API token, API key, OpenAI key, and username.
            """        
        return {
            "api_token": self.headers['Authorization'],
            "api_key": self.headers['x-api-key'],
            "openai_key": self.headers['openai-key'],
            "username": self.username
        }
