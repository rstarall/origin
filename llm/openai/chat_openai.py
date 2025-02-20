from abc import ABC, abstractmethod
import openai
from llm.chat_llm import ChatLLM


class ChatOpenAI(ChatLLM):
    """
    ChatOpenAI implements the ChatLLM abstract class for OpenAI's GPT models.
    """

    def __init__(self, api_key: str, uri: str = None, model: str = "gpt-4o"):
        self.api_key = api_key
        self.uri = uri
        self.model = model
        self.config = {"temperature": 0.7, "max_tokens": 256}
        
        # Create client configuration
        client_params = {
            "api_key": self.api_key,
            "timeout": 60.0
        }
        # Add custom URI to client configuration if provided
        if self.uri:
            client_params["base_url"] = self.uri
            
        self.client = openai.OpenAI(**client_params)

    def input(self, query: str, **kwargs) -> str:
        """
        Processes a user's query and returns the model's response.
        Compatible with OpenAI Python SDK >=1.0.0.
        """
        params = {**self.config, **kwargs}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                **params,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API request failed: {str(e)}")

    def stream_output(self, query: str, **kwargs):
        """
        Streams the model's response step-by-step for a given query.
        Compatible with OpenAI Python SDK >=1.0.0.
        """
        params = {**self.config, **kwargs}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": query}],
                stream=True,
                **params,
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise RuntimeError(f"API stream request failed: {str(e)}")

    def configure(self, **kwargs):
        """
        Configures model parameters, such as `temperature` or `max_tokens`.
        """
        self.config.update(kwargs)

    def reset(self):
        """
        Resets the model's configuration settings to their default state.
        """
        self.config = {"temperature": 0.7, "max_tokens": 256}