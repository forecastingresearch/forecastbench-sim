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

# langchain/openai/pinecone removed in the Gemini port.
from civrealm.freeciv.utils.freeciv_logging import fc_logger

from agents.prompt_handlers.base_prompt_handler import BasePromptHandler

from .base_worker import BaseWorker
from .gpt_worker import AzureGPTWorker


class AdvisorWorker(AzureGPTWorker):
    """
    This worker does not control any enitity, but provides suggestions
    to workers controlling one.
    """
