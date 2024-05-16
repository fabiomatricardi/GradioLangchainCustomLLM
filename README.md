# GradioLangchainCustomLLM
How to build a Custom LLM wrapping a Gradio API call into Langchain


Full explanation in [Medium article here](https://medium.com/generative-ai/i-hacked-the-ai-agents-now-you-can-have-them-all-for-free-08cae7d29714)
---

> What I'm seeing with AI agents is an exciting trend that I believe everyone building AI should pay attention to…

*Andrew Ng, Co-founder of Google Brain, former Chief Scientist at Baidu, founder of Coursera*

---

Few weeks ago I discovered a secret hack to use for free Gradio API call (you can read more here and here). I also completed my study of the amazing book by Ben Auffarth about Langchain in AI… and I got inspired.
I was wandering if there is a way to use Gradio API with Langchain so that we can test for free the endless possibilities of AI agents. Are you ready to lie the foundations for free agents working for you?

We need to create a Custom Wrapper
Langchain has a huge collection of integrations: basically you can connect to Language Models, Document Loaders, Databases and much more in an modular and easy way. 
Thankfully they kept open the possibility to create Custom LLM classes that can be used with all their toolset.

[Custom LLM | 🦜️🔗 LangChain](https://python.langchain.com/v0.1/docs/modules/model_io/llms/custom_llm/)

Here I learned the process to how to create a custom LLM wrapper, in case you want to use your own LLM or a different wrapper than one that is supported in LangChain.
Wrapping your LLM with the standard LLM interface allow you to use your LLM in existing LangChain programs with minimal code modifications!
As an bonus, your LLM will automatically become a LangChain Runnable and will benefit from some optimizations out of the box, async support, the astream_events API, etc.
So let's do it. In this example we will connect Langchain to Llama-3–8b, but the process is the same (with few little tricks) for all the others Gradio API on the Hugging Face Hub Demo applications.


Google Colab Notebook 📚 [here](https://github.com/fabiomatricardi/GradioLangchainCustomLLM/raw/main/Gradio%2BLangChain%3DFreeAI_Agents.ipynb)


### 🌟 The results:

<img src='https://github.com/fabiomatricardi/GradioLangchainCustomLLM/raw/main/GradiolangChainStreaming.gif' width=900>


### 💻 The Code

```
%pip install --upgrade --quiet  gradio_tools huggingface_hub langchain

from typing import Any, Dict, Iterator, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class GradioClientChat(LLM):
    """
    Custom LLM class based on the Gradio API call.
    """
    from gradio_client import Client
    chatbot: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Instantiating the ChatBot class
        # add here you hf_token, in case as shown here below
        #yourHFtoken = "hf_xxxxxxxxxxxxxxxxx" #here your HF token
        #self.chatbot =("ysharma/Chat_with_Meta_llama3_8b", hf_token=yourHFtoken)
        self.chatbot = Client("ysharma/Chat_with_Meta_llama3_8b")

    @property
    def _llm_type(self) -> str:
        return "Gradio API client Meta_llama3_8b"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            chatbot=None,
            request: float = 0.95,
            param: float = 512,
    ) -> str:
        """
        Make an API call to the Gradio API client Meta_llama3_8b using the specified prompt and return the response.
        """
        if chatbot is None:
            chatbot = self.chatbot

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # Return the response from the API
        result = chatbot.predict(   #.submit for streaming effect / .predict for normal output
            		message=prompt,
                request=request,
                param_3=param,
                api_name="/chat"
        )
        return str(result)

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        chatbot=None,
        request: float = 0.95,
        param: float = 512,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
        if chatbot is None:
            chatbot = self.chatbot

        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # Return the response from the API
        for char in chatbot.submit(   #.submit for streaming effect / .predict for normal output
            		message=prompt,
                request=request,
                param_3=param,
                api_name="/chat"
                ):
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk
```

### ⚠️ To run the inference:

```
llm = GradioClientChat()

# INference with no parameters
result = llm.invoke("what is artificial Intelligence?")  #[10:]   to remove the assitant from the output
print(result)

# inference with temperature and ma_lenght
result = llm.invoke("what are the differences between artificial Intelligence and machine learning?", request = 0.45, param = 600)[10:]  # to remove the assitant from the output
print(result)
```


### 🥂 To run the inference wit streaming effect:

```
final = ''
for token in llm.stream("what is the scientific method?",request = 0.25, param = 600):
        if final == '':
            final=token
            print(token, end="", flush=True)
        else:
            try:
                print(token.replace(final,''), end="", flush=True)
                final = token
            except:
                pass
```

