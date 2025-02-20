# Import required modules
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from llm.openai.chat_openai import ChatOpenAI


def test_chat_openai_response(api_key:str,uri:str = None):
    """
    Test whether the input method of the ChatOpenAI class can return a response properly.
    """
    # Initialize an instance of the ChatOpenAI class with optional custom URI
    chat = ChatOpenAI(api_key=api_key, model="gpt-4o", uri=uri)

    # Define an input message
    query = "who are you?"

    # Test the input method and print the response
    response = chat.input(query)
    print("Model Response:", response)


def test_chat_openai_stream(api_key:str,uri:str = None):
    """
    Test whether the stream_output method of the ChatOpenAI class can return a streaming response.
    """
    # Initialize an instance of the ChatOpenAI class with optional custom URI
    chat = ChatOpenAI(api_key=api_key, model="gpt-4o", uri=uri)

    # Define an input message
    query = "who are you?please output by json formation.Do not include any other character besides json"

    # Stream the response and print it
    print("Stream Response:")
    for chunk in chat.stream_output(query):
        print(chunk, end="")


def test_chat_openai_with_custom_uri():
    """
    Test the ChatOpenAI class with a custom OpenAI source URI.
    """
    api_key = ""
    custom_uri = "https://chatapi.littlewheat.com/v1"  # Replace with your custom OpenAI URI
    print(f"Testing with custom OpenAI URI: {custom_uri}")

    # Call the test functions with the custom URI
    test_chat_openai_response(api_key=api_key,uri=custom_uri)
    test_chat_openai_stream(api_key=api_key,uri=custom_uri)


if __name__ == "__main__":
    print("\nTesting with custom OpenAI source...")
    test_chat_openai_with_custom_uri()