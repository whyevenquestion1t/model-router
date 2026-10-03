"""A LangChain agent middleware that routes each thread to a model tier.

Classifies a thread's first human message once, picks the cheapest model
tier likely to complete the task, and reuses that choice for the rest of
the thread. See the README for background and usage.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Literal

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from pydantic import BaseModel, create_model
from typing_extensions import NotRequired

from .tiers import ModelTier

logger = logging.getLogger(__name__)

DEFAULT_BASE_PROMPT = (
    "You are a routing classifier for an AI agent. Read the user's request "
    "and choose the CHEAPEST model tier that is still likely to complete "
    "the task successfully. Do not pick a more capable tier than the task "
    "needs."
)


class ModelRouterState(AgentState):
    """Agent state extended with the thread's cached tier decision."""

    selected_tier: NotRequired[str]


class ModelRouterMiddleware(AgentMiddleware):
    """Routes a thread to a model tier once, then reuses that choice.

    On the first model call of a thread, a classifier model reads the
    first human message and the per-tier criteria you provide, and picks a
    tier. The choice is cached on agent state (``selected_tier``), so every
    later turn in the same thread skips classification and reuses it.

    Example:
        >>> from langchain.chat_models import init_chat_model
        >>> from model_router import ModelRouterMiddleware, ModelTier
        >>> tiers = [
        ...     ModelTier("fast", init_chat_model("claude-haiku-4-5"), "Trivial lookups, formatting, simple Q&A."),
        ...     ModelTier("balanced", init_chat_model("claude-sonnet-4-6"), "Everyday feature work and bug fixes."),
        ...     ModelTier("performance", init_chat_model("claude-opus-4-6"), "Multi-file refactors, tricky debugging."),
        ... ]
        >>> router = ModelRouterMiddleware(
        ...     tiers=tiers,
        ...     classifier_model=init_chat_model("claude-haiku-4-5"),
        ...     default_tier="balanced",
        ... )
    """

    state_schema = ModelRouterState

    def __init__(
        self,
        tiers: Sequence[ModelTier],
        classifier_model: BaseChatModel,
        *,
        default_tier: str | None = None,
        base_prompt: str = DEFAULT_BASE_PROMPT,
    ) -> None:
        """Configure the router.

        Args:
            tiers: The model tiers to route between, in any order. Must
                have unique names and at least one entry.
            classifier_model: The chat model used to classify requests.
                It does not need to be one of the tiers' models — a small,
                cheap model is usually the right choice here.
            default_tier: Tier to fall back to if classification fails or
                returns something unexpected. Defaults to the first tier
                in ``tiers``.
            base_prompt: Overrides the instruction given to the
                classifier. Keep the "pick the cheapest tier that can do
                the job" framing unless you have a reason not to.
        """
        super().__init__()
        if not tiers:
            raise ValueError("ModelRouterMiddleware needs at least one tier")
        self.tiers = {tier.name: tier for tier in tiers}
        if len(self.tiers) != len(tiers):
            raise ValueError("tier names must be unique")

        self.default_tier = default_tier if default_tier is not None else tiers[0].name
        if self.default_tier not in self.tiers:
            raise ValueError(
                f"default_tier {self.default_tier!r} is not one of the configured tiers: "
                f"{sorted(self.tiers)}"
            )

        self.classifier_model = classifier_model
        self.base_prompt = base_prompt
        self._classifier = classifier_model.with_structured_output(self._build_schema())

    def _build_schema(self) -> type[BaseModel]:
        """Build a structured-output schema restricted to this router's tier names."""
        names = tuple(self.tiers)
        return create_model("TierSelection", tier=(Literal[names], ...))  # type: ignore[valid-type]

    @staticmethod
    def _first_human_message(state: ModelRouterState) -> str:
        for message in state["messages"]:
            if isinstance(message, HumanMessage):
                content = message.content
                return content if isinstance(content, str) else str(content)
        return ""

    def _build_prompt(self, request_text: str) -> str:
        criteria = "\n".join(f"- {name}: {tier.criteria}" for name, tier in self.tiers.items())
        return f"{self.base_prompt}\n\nAvailable tiers:\n{criteria}\n\nUser request:\n{request_text}"

    def _log_classification_failure(self) -> None:
        logger.warning(
            "model_router: classification failed, falling back to default tier %r",
            self.default_tier,
            exc_info=True,
        )

    def _classify(self, request_text: str) -> str:
        try:
            result = self._classifier.invoke(self._build_prompt(request_text))
            return result.tier  # type: ignore[attr-defined]
        except Exception:
            self._log_classification_failure()
            return self.default_tier

    async def _aclassify(self, request_text: str) -> str:
        try:
            result = await self._classifier.ainvoke(self._build_prompt(request_text))
            return result.tier  # type: ignore[attr-defined]
        except Exception:
            self._log_classification_failure()
            return self.default_tier

    def _resolve_tier_name(self, tier_name: str) -> str:
        return tier_name if tier_name in self.tiers else self.default_tier

    def before_model(self, state: ModelRouterState, runtime: Runtime) -> dict[str, Any] | None:
        """Classify the thread once and cache the result on state."""
        if state.get("selected_tier"):
            return None

        request_text = self._first_human_message(state)
        tier_name = self._classify(request_text) if request_text else self.default_tier
        return {"selected_tier": self._resolve_tier_name(tier_name)}

    async def abefore_model(self, state: ModelRouterState, runtime: Runtime) -> dict[str, Any] | None:
        """Async counterpart of `before_model`."""
        if state.get("selected_tier"):
            return None

        request_text = self._first_human_message(state)
        tier_name = await self._aclassify(request_text) if request_text else self.default_tier
        return {"selected_tier": self._resolve_tier_name(tier_name)}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Swap in the thread's selected tier's model for this call."""
        tier_name = request.state.get("selected_tier", self.default_tier)
        tier = self.tiers.get(tier_name, self.tiers[self.default_tier])
        return handler(request.override(model=tier.model))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async counterpart of `wrap_model_call`."""
        tier_name = request.state.get("selected_tier", self.default_tier)
        tier = self.tiers.get(tier_name, self.tiers[self.default_tier])
        return await handler(request.override(model=tier.model))
