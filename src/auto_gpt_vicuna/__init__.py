"""This is the Vicuna plugin for Auto-GPT."""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar

from auto_vicuna.__main__ import load_model
from auto_vicuna.chat import chat_one_shot
from auto_vicuna.conversation import make_conversation
from auto_gpt_plugin_template import AutoGPTPluginTemplate

import torch


PromptGenerator = TypeVar("PromptGenerator")


class AutoGPTPVicuna(AutoGPTPluginTemplate):
    """
    This is the Vicuna local model  plugin for Auto-GPT.
    """

    def __init__(self):
        super().__init__()
        self._name = "Auto-GPT-Vicuna"
        self._version = "0.1.0"
        self._description = "This is a Vicuna local model plugin."
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vicuna_weights = os.environ.get("VICUNA_WEIGHTS", "")
        self.load_8bit = os.environ.get("LOAD_8BIT", False)

        model, tokenizer = load_model(
            self.vicuna_weights,
            device=self.device,
            num_gpus=1,
            debug=False,
            load_8bit=self.load_8bit,
        )
        self.model = model
        self.tokenizer = tokenizer

        model.eval()

    def can_handle_on_response(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_response method.

        Returns:
            bool: True if the plugin can handle the on_response method."""
        return False

    def on_response(self, response: str, *args, **kwargs) -> str:
        """This method is called when a response is received from the model."""
        pass

    def can_handle_post_prompt(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_prompt method.

        Returns:
            bool: True if the plugin can handle the post_prompt method."""
        return True

    def post_prompt(self, prompt: PromptGenerator) -> PromptGenerator:
        """This method is called just after the generate_prompt is called,
            but actually before the prompt is generated.

        Args:
            prompt (PromptGenerator): The prompt generator.

        Returns:
            PromptGenerator: The prompt generator.
        """

        prompt.add_constraint(
            "All output must be in format that can be read by Python json.loads, always"
            " ensure output is in the proper format so it can be read."
    	 )
        prompt.add_constraint(
            f"""Always use the following format for your responses:

{{{
    'command': {{{
        'name': 'command name',
        'args': {{{
            'arg name': 'value'
        }}}
    }}},
    'thoughts':
    {{{
        'text': 'thought',
        'reasoning': 'reasoning',
        'plan': '- short bulleted\n- list that conveys\n- long-term plan',
        'criticism': 'constructive self-criticism',
        'speak': 'thoughts summary to say to user'
    }}}
}}}
"""
        )
    
        #prompt.add_constraint(
            #"~4000 word limit for short term memory. Your short term memory is short, so"
            #" immediately save important information to files."
       	 #)
        #prompt.add_constraint(
          #"If you are unsure how you previously did something or want to recall past"
          #" events, thinking about similar events will help you remember."
       #)
        #prompt.add_constraint("No user assistance")
        #prompt.add_constraint(
          #'Exclusively use the commands listed in double quotes e.g. "command name"'
         #)
  
      # Define the command list
        #commands = [
          #("Do Nothing", "do_nothing", {}),
          #("Task Complete (Shutdown)", "task_complete", {"reason": "<reason>"}),
    #("Analyze Code", "analyze_code", {"code": "<full_code_string>"}),
    #("Execute Code", "execute_code", {"code": "<full_code_string>"}),
    #("File Operations", "file_operations", {}),
    #("Google Search", "google_search", {"query": "<query>"}),
    #("Improve Code", "improve_code", {"suggestions": "<list_of_suggestions>", "code": "<full_code_string>"}),
    #("Web Requests", "web_requests", {}),
    #("Web Selenium", "web_selenium", {"url": "<url>", "question": "<what_you_want_to_find_on_website>"})
        #]
  
      # Add commands to the PromptGenerator object
        #for command_label, command_name, args in commands:
          #prompt.add_command(command_label, command_name, args)
  
      # Add resources to the PromptGenerator object
        #prompt.add_resource(
          #"Internet access for searches and information gathering."
        #)
        #prompt.add_resource("Long Term memory management.")
        #prompt.add_resource(
          #"Vicuna powered Agents for delegation of simple tasks."
        #)
        #prompt.add_resource("File output.")
  
      # Add performance evaluations to the PromptGenerator object
        #prompt.add_performance_evaluation(
          #"Continuously review and analyze your actions to ensure you are performing to"
          #" the best of your abilities."
        #)
        #prompt.add_performance_evaluation(
          #"Constructively self-criticize your big-picture behavior constantly."
        #)
        #prompt.add_performance_evaluation(
          #"Reflect on past decisions and strategies to refine your approach."
        #)
        #prompt.add_performance_evaluation(
          #"Every command has a cost, so be smart and efficient. Aim to complete tasks in"
          #" the least number of steps."
        #)
        #prompt.add_performance_evaluation("Write all code to a file.")
        
        return prompt
        


    def can_handle_on_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_planning method.

        Returns:
            bool: True if the plugin can handle the on_planning method."""
        return False

    def on_planning(
        self, prompt: PromptGenerator, messages: List[str]
    ) -> Optional[str]:
        """This method is called before the planning chat completeion is done.

        Args:
            prompt (PromptGenerator): The prompt generator.
            messages (List[str]): The list of messages.
        """
        pass

    def can_handle_post_planning(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_planning method.

        Returns:
            bool: True if the plugin can handle the post_planning method."""
        return False

    def post_planning(self, response: str) -> str:
        """This method is called after the planning chat completeion is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_pre_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_instruction method.

        Returns:
            bool: True if the plugin can handle the pre_instruction method."""
        return False

    def pre_instruction(self, messages: List[str]) -> List[str]:
        """This method is called before the instruction chat is done.

        Args:
            messages (List[str]): The list of context messages.

        Returns:
            List[str]: The resulting list of messages.
        """
        pass

    def can_handle_on_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the on_instruction method.

        Returns:
            bool: True if the plugin can handle the on_instruction method."""
        return False

    def on_instruction(self, messages: List[str]) -> Optional[str]:
        """This method is called when the instruction chat is done.

        Args:
            messages (List[str]): The list of context messages.

        Returns:
            Optional[str]: The resulting message.
        """
        pass

    def can_handle_post_instruction(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_instruction method.

        Returns:
            bool: True if the plugin can handle the post_instruction method."""
        return False

    def post_instruction(self, response: str) -> str:
        """This method is called after the instruction chat is done.

        Args:
            response (str): The response.

        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_pre_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the pre_command method.

        Returns:
            bool: True if the plugin can handle the pre_command method."""
        return False

    def pre_command(
        self, command_name: str, arguments: Dict[str, Any]
    ) -> Tuple[str, Dict[str, Any]]:
        """This method is called before the command is executed.

        Args:
            command_name (str): The command name.
            arguments (Dict[str, Any]): The arguments.

        Returns:
            Tuple[str, Dict[str, Any]]: The command name and the arguments.
        """
        pass

    def can_handle_post_command(self) -> bool:
        """This method is called to check that the plugin can
        handle the post_command method.

        Returns:
            bool: True if the plugin can handle the post_command method."""
        return False

    def post_command(self, command_name: str, response: str) -> str:
        """This method is called after the command is executed.

        Args:
            command_name (str): The command name.
            response (str): The response.

        Returns:
            str: The resulting response.
        """
        pass

    def can_handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        """This method is called to check that the plugin can
          handle the chat_completion method.

        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

          Returns:
              bool: True if the plugin can handle the chat_completion method."""
        return True

    def handle_chat_completion(
        self,
        messages: list[Dict[Any, Any]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """This method is called when the chat completion is done.

        Args:
            messages (Dict[Any, Any]): The messages.
            model (str): The model name.
            temperature (float): The temperature.
            max_tokens (int): The max tokens.

        Returns:
            str: The resulting response."""
        
        roles = {message["role"] for message in messages}
        last_message = messages.pop()["content"]
        conv = make_conversation(
            "",
            list(roles),
            [(message["role"], message["content"]) for message in messages],
        )
        if max_tokens is None:
            max_tokens = 2048
        max_tokens = min(max_tokens, 2048)
        with torch.inference_mode():
            return chat_one_shot(
                self.model,
                self.tokenizer,
                self.vicuna_weights,
                self.device,
                conv,
                last_message,
                temperature,
                max_tokens,
            )
