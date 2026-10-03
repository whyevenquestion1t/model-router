"""model_router: a LangChain agent middleware that routes tasks to model tiers by cost."""

from .middleware import DEFAULT_BASE_PROMPT, ModelRouterMiddleware, ModelRouterState
from .tiers import ModelTier

__all__ = [
    "DEFAULT_BASE_PROMPT",
    "ModelRouterMiddleware",
    "ModelRouterState",
    "ModelTier",
]

__version__ = "0.1.0"
