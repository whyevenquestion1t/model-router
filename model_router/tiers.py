"""Model tier definitions for the router."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel


@dataclass(frozen=True)
class ModelTier:
    """A single rung on the router's cost/intelligence ladder.

    Attributes:
        name: Short identifier used in prompts and as the classifier's
            structured-output value (e.g. ``"fast"``, ``"balanced"``,
            ``"performance"``). Must be unique within a router.
        model: Any LangChain chat model instance (e.g. from
            ``init_chat_model``). The router is provider-agnostic: tiers
            can mix models from different providers.
        criteria: A short, plain-language description of the kind of task
            that belongs on this tier. This is shown to the classifier
            model, so write it the way you'd brief a teammate.
    """

    name: str
    model: BaseChatModel
    criteria: str
