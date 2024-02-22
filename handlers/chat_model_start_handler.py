from langchain.callbacks.base import BaseCallbackHandler
from pyboxen import boxen


def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))


# boxen_print("TEXT HERE!", title="human", color="red")


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n\n=============== Sending Messages ==================\n\n")
        for message in messages[0]:
            # system message
            if message.type == "system":
                boxen_print(message.content, title=message.type, color="yellow")
            # user message
            elif message.type == "human":
                boxen_print(message.content, title=message.type, color="green")
            # ai is trying to run a function
            elif message.type == "ai" and "function_call" in message.additional_kwargs:
                call = message.additional_kwargs["function_call"]
                boxen_print(
                    f"Running tool {call['name']} with args: {call['arguments']}",
                    title=message.type,
                    color="cyan",
                )
            # this is the output of ai
            elif message.type == "ai":
                boxen_print(message.content, title=message.type, color="blue")
            # result of function call
            elif message.type == "function":
                boxen_print(message.content, title=message.type, color="purple")
            # all other cases
            else:
                boxen_print(message.content, title=message.type, color="white")
