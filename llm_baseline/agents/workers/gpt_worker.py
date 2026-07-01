# Copyright (C) 2023  The CivRealm project
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import random
import json
import time

# Ported to Gemini via our litellm wrapper. The original Azure-OpenAI / langchain /
# Pinecone stack is stubbed (see docs/forecast_uplift_study.md). RAG (Pinecone) and
# langchain conversation memory are NOT on the action-generation critical path.
from civrealm.evaluation.models import LiteLLMModel

from civrealm.freeciv.utils.freeciv_logging import fc_logger
from agents.prompt_handlers.base_prompt_handler import BasePromptHandler

from .base_worker import BaseWorker

# Model id is configurable so the run script can pick it (default: Gemini 2.5 Flash).
UPLIFT_MODEL_ID = os.environ.get("UPLIFT_MODEL", "google/gemini-2.5-flash")


class _StubMemory:
    """Drop-in for langchain ConversationSummaryBufferMemory. We don't summarize;
    per-actor dialogues are short (one actor, one turn)."""

    def save_context(self, *_args, **_kwargs):
        pass

    def load_memory_variables(self, _):
        return {"history": ""}


class AzureGPTWorker(BaseWorker):
    """
    This agent uses GPT-3 to generate actions.
    """
    def __init__(self,
                 model: str = UPLIFT_MODEL_ID,
                 prompt_prefix: str = "civ_prompts",
                 **kwargs):
        self.prompt_prefix = prompt_prefix
        super().__init__(model, **kwargs)

    def init_prompts(self):
        self.prompt_handler = BasePromptHandler(
            prompt_prefix=self.prompt_prefix)
        self._load_instruction_prompt()
        self._load_task_prompt()

    def init_llm(self):
        self.llm = LiteLLMModel(id=self.model)
        self.chain = None            # RAG disabled
        self.memory = _StubMemory()  # no langchain memory

    def init_index(self):
        self.index = None            # Pinecone disabled

    def _load_instruction_prompt(self):
        instruction_prompt = self.prompt_handler.instruction_prompt()
        self.add_user_message_to_dialogue(instruction_prompt)

    def _load_task_prompt(self):
        task_prompt = self.prompt_handler.task_prompt()
        self.add_user_message_to_dialogue(task_prompt)

    def register_all_commands(self):
        self.register_command('manualAndHistorySearch',
                              self.handle_command_manual_and_history_search)
        self.register_command('finalDecision',
                              self.handle_command_final_decision)
        self.register_command('suggestion', self.handle_command_suggestion)

    def handle_command_manual_and_history_search(self, command_input,
                                                 obs_input_prompt,
                                                 current_avail_actions):
        # RAG (Pinecone) disabled: never consult the index. Nudge the model to
        # make a decision instead of looking things up.
        answer = self.prompt_handler.finish_look_for()
        self.add_user_message_to_dialogue(answer)
        self.taken_actions_list.append('look_up')
        return None, ''

    def handle_command_ask_current_game_information(self, command_input,
                                                    obs_input_prompt,
                                                    current_avail_actions):
        self.taken_actions_list.append('askCurrentGameInformation')
        return None, ''

    def handle_command_suggestion(self, command_input, obs_input_prompt,
                                  current_avail_actions):
        exec_action = command_input["suggestion"]
        return exec_action, ''

    def handle_command_final_decision(self, command_input, obs_input_prompt,
                                      current_avail_actions):
        exec_action = command_input['action']
        lower_avail_actions = [x.lower() for x in current_avail_actions]
        if exec_action.lower() not in lower_avail_actions:
            print(f'{self.name}\'s chosen action "{exec_action}" not in the ' +
                  f'available action list, available actions are ' +
                  f'{current_avail_actions}, retrying...')
            fc_logger.error(
                f'{self.name}\'s chosen action "{exec_action}" not in the '
                f'available action list {current_avail_actions}, retrying...')
            return None, self.prompt_handler.insist_avail_action()

        self.taken_actions_list.append(command_input['action'])

        for move_name in current_avail_actions:
            if move_name[:4] != "move":
                continue
            if self.taken_actions_list_needs_update(move_name, 15, 4):
                return None, self.prompt_handler.insist_various_actions(
                    action=move_name)

        return exec_action, ''

    def query_llm(self, stop=None, temperature=0.7, top_p=0.95):
        fc_logger.debug(f'Querying with dialogue: {self.dialogue}')
        # Flatten the chat dialogue into a single prompt for our litellm wrapper,
        # then re-wrap the plain-text reply into the {choices:[{message:...}]}
        # shape that parse_response()/choose_action() expect.
        prompt = "\n\n".join(
            f"[{m.get('role', 'user')}]\n{m.get('content', '')}"
            for m in self.dialogue
        )
        # No max_tokens: never truncate the model's output (Gemini 2.5 also
        # spends tokens on hidden thinking, which truncation would starve).
        text = self.llm.get_response(prompt, temperature=temperature,
                                     max_tokens=None)
        return {"choices": [{"message": {"role": "assistant",
                                         "content": text or ""}}]}

    def generate_command(self, prompt: str):
        self.add_user_message_to_dialogue(prompt +
                                          self.prompt_handler.insist_json())
        self.restrict_dialogue()
        response = self.query_llm()
        self.memory.save_context({'user': prompt},
                                 {'assistant': str(response)})
        return response

    def parse_response(self, response):
        content = response['choices'][0]['message']['content']
        # Always extract the {...} span first — models often wrap JSON in ```json
        # fences or add prose, which broke the original (it only sliced when the
        # braces were unbalanced, so fenced-but-balanced JSON failed to parse).
        start_index = content.find('{')
        end_index = content.rfind('}') + 1
        if start_index != -1 and end_index > start_index:
            content = content[start_index:end_index]
        rlack = content.count("{") - content.count("}")
        if rlack > 0:
            content = content + "}" * rlack
        return json.loads(content)

    def process_command(self, response, obs_input_prompt,
                        current_avail_actions):
        # First try to parse the reponse by the given json format
        fc_logger.debug(f'Processing response: {response}')
        _dbg = os.environ.get("DEBUG_AGENT")
        try:
            command_json = self.parse_response(response)
            command_input = command_json['command']['input']
            command_name = command_json['command']['name']
        except Exception as e:
            fc_logger.error(
                f'\nRESPONSE:{response}\nCommond json parsing error: {e}')
            if _dbg:
                content = response['choices'][0]['message']['content']
                print(f'[DBG parse-fail: {e}] content[:200]={content[:200]!r}', flush=True)
            print('Not in given json format, retrying...')
            return None, self.prompt_handler.insist_json()

        if _dbg:
            print(f'[DBG command={command_name} input={str(command_input)[:80]}]', flush=True)

        # Then check if the command is valid
        if command_name not in self.command_handlers:
            fc_logger.error(f'Unknown command: {command_name}')
            available_commands = ', '.join(self.command_handlers.keys())
            prompt_addition = self.prompt_handler.insist_available_commands(
                available_commands)
            return None, prompt_addition

        return self.command_handlers[command_name](command_input,
                                                   obs_input_prompt,
                                                   current_avail_actions)
