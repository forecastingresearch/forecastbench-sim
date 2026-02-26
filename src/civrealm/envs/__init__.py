try:
    from .parallel_tensor_env import ParallelTensorEnv
except (ImportError, AttributeError):
    ParallelTensorEnv = None
from .freeciv_minitask_env import FreecivMinitaskEnv
from .freeciv_base_env import FreecivBaseEnv
from .freeciv_tensor_env import FreecivTensorEnv
from .freeciv_tensor_minitask_env import FreecivTensorMinitaskEnv


from .freeciv_llm_env import FreecivLLMEnv
# Parallel environment
try:
    from .freeciv_parallel_env import FreecivParallelEnv
except (ImportError, AttributeError):
    FreecivParallelEnv = None
try:
    from .freeciv_a3c_env import FreecivA3CEnv
except (ImportError, AttributeError):
    FreecivA3CEnv = None
# Minitask environment
