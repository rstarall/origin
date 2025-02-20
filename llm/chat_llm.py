from abc import ABC, abstractmethod


class ChatLLM(ABC):
    """
    Abstract base class for defining a standard interface for language models.
    """

    @abstractmethod
    def input(self, query: str, **kwargs) -> str:
        """User input method, accepts a query and returns the model's full response.
        
        Parameters:
            query: The user's query as a string.
            kwargs: Optional additional parameters.
        Returns:
            The model's response as a string.
        """
        pass

    @abstractmethod
    def stream_output(self, query: str, **kwargs):
        """Stream output method, processes a query and generates the response gradually
        (e.g., step-by-step output for generating long text).
        
        Parameters:
            query: The user's query as a string.
            kwargs: Optional additional parameters.
        Yields:
            A step-by-step response from the model.
        """
        pass

    @abstractmethod
    def configure(self, **kwargs):
        """Configure method for setting model parameters, e.g., hyperparameters or API keys.
        
        Parameters:
            kwargs: Configuration options.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset method for clearing context or state from the model.
        """
        pass


# Example: Subclass implementation
class ExampleChatLLM(ChatLLM):
    def __init__(self):
        self.config = {}

    def input(self, query: str, **kwargs) -> str:
        # Simulate returning a full response from the model
        return f"Echo: {query}"

    def stream_output(self, query: str, **kwargs):
        # Simulate streaming output word-by-word
        for word in query.split():
            yield word

    def configure(self, **kwargs):
        # Update the configuration dictionary
        self.config.update(kwargs)

    def reset(self):
        # Reset the configuration to an empty state
        self.config = {}