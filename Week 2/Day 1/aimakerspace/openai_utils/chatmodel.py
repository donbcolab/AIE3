from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from a .env file
load_dotenv()

class ChatOpenAI:
    """
    A class to interact with OpenAI's GPT-3.5-turbo-0125 model for chat completions.

    Attributes:
        model_name (str): The name of the model to use.
        openai_api_key (str): The OpenAI API key.
        default_seed (int): The default seed for reproducibility.
        default_temperature (float): The default temperature for deterministic output.

    Methods:
        run(input_messages, text_only=True, **kwargs): Generates a response from the model based on the input messages.
    """

    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", default_seed: int = 42, default_temperature: float = 0.2):
        """
        Initializes the ChatOpenAI class with the specified model name, default seed, and default temperature.

        Args:
            model_name (str, optional): The name of the model to use. Defaults to "gpt-3.5-turbo-0125".
            default_seed (int, optional): The default seed for reproducibility. Defaults to 42.
            default_temperature (float, optional): The default temperature for deterministic output. Defaults to 0.2.

        Raises:
            ValueError: If the OPENAI_API_KEY environment variable is not set.
        """
        self.model_name = model_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.default_seed = default_seed
        self.default_temperature = default_temperature
        if self.openai_api_key is None:
            raise ValueError("OPENAI_API_KEY is not set")

    def run(self, input_messages, text_only: bool = True, **kwargs):
        """
        Generates a response from the model based on the input messages.

        Args:
            input_messages (list): A list of messages to be sent to the model.
            text_only (bool, optional): If True, returns only the text content of the response. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the OpenAI API call. Overrides default temperature and seed if provided.

        Returns:
            str or dict: The response from the model. If text_only is True, returns the text content of the response.
                         Otherwise, returns the full response object.

        Raises:
            ValueError: If input_messages is not a list.
        """
        if not isinstance(input_messages, list):
            raise ValueError("input_messages must be a list")

        client = OpenAI()
        
        # Set default temperature and seed if not provided
        temperature = kwargs.pop('temperature', self.default_temperature)
        seed = kwargs.pop('seed', self.default_seed)

        api_call_params = {
            'model': self.model_name,
            'messages': input_messages,
            'temperature': temperature,
            'seed': seed,
        }

        # Add remaining kwargs to the parameters
        api_call_params.update(kwargs)

        response = client.chat.completions.create(**api_call_params)

        if text_only:
            return response.choices[0].message.content

        return response

if __name__ == "__main__":
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-0125")
    main_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What are some tips for being an effective CEO?"}
    ]
    response = chat.run(main_messages, max_tokens=100)
    print("Response:", response)
