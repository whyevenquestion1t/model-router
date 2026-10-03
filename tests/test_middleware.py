from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from model_router import ModelRouterMiddleware, ModelTier


class _FakeStructuredClassifier:
    """Stands in for `classifier_model.with_structured_output(schema)`."""

    def __init__(self, tier: str | None = None, error: Exception | None = None):
        self._tier = tier
        self._error = error

    def invoke(self, prompt: str):
        if self._error is not None:
            raise self._error
        return SimpleNamespace(tier=self._tier)

    async def ainvoke(self, prompt: str):
        return self.invoke(prompt)


class FakeClassifierModel:
    """Stands in for a LangChain chat model, for classifier_model."""

    def __init__(self, tier: str | None = None, error: Exception | None = None):
        self._structured = _FakeStructuredClassifier(tier=tier, error=error)

    def with_structured_output(self, schema):
        return self._structured


class FakeModelRequest:
    """Stands in for ModelRequest: carries state and records overrides."""

    def __init__(self, state: dict, model=None):
        self.state = state
        self.model = model

    def override(self, model):
        return FakeModelRequest(state=self.state, model=model)


def make_tiers():
    return [
        ModelTier("fast", model="fast-model", criteria="Trivial lookups and formatting."),
        ModelTier("balanced", model="balanced-model", criteria="Everyday feature work."),
        ModelTier("performance", model="performance-model", criteria="Tricky multi-file debugging."),
    ]


def test_classifies_first_human_message_and_caches():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(tier="fast"),
        default_tier="balanced",
    )
    state = {"messages": [HumanMessage(content="rename this variable")]}

    update = router.before_model(state, runtime=None)
    assert update == {"selected_tier": "fast"}

    state["selected_tier"] = "fast"
    assert router.before_model(state, runtime=None) is None


def test_falls_back_to_default_when_classifier_errors():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(error=RuntimeError("boom")),
        default_tier="balanced",
    )
    state = {"messages": [HumanMessage(content="fix the flaky test")]}

    assert router.before_model(state, runtime=None) == {"selected_tier": "balanced"}


def test_falls_back_to_default_when_classifier_returns_unknown_tier():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(tier="super-ultra"),
        default_tier="balanced",
    )
    state = {"messages": [HumanMessage(content="anything")]}

    assert router.before_model(state, runtime=None) == {"selected_tier": "balanced"}


def test_uses_default_tier_without_classifying_when_no_human_message():
    classifier = FakeClassifierModel(tier="fast")
    router = ModelRouterMiddleware(tiers=make_tiers(), classifier_model=classifier, default_tier="performance")
    state = {"messages": [AIMessage(content="hello")]}

    assert router.before_model(state, runtime=None) == {"selected_tier": "performance"}


def test_wrap_model_call_overrides_model_for_selected_tier():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(tier="fast"),
        default_tier="balanced",
    )
    request = FakeModelRequest(state={"selected_tier": "performance"})

    result = router.wrap_model_call(request, handler=lambda req: req.model)

    assert result == "performance-model"


def test_wrap_model_call_falls_back_to_default_when_no_tier_selected():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(tier="fast"),
        default_tier="balanced",
    )
    request = FakeModelRequest(state={})

    result = router.wrap_model_call(request, handler=lambda req: req.model)

    assert result == "balanced-model"


def test_requires_at_least_one_tier():
    with pytest.raises(ValueError, match="at least one tier"):
        ModelRouterMiddleware(tiers=[], classifier_model=FakeClassifierModel())


def test_rejects_duplicate_tier_names():
    tiers = [
        ModelTier("fast", model="a", criteria="x"),
        ModelTier("fast", model="b", criteria="y"),
    ]
    with pytest.raises(ValueError, match="unique"):
        ModelRouterMiddleware(tiers=tiers, classifier_model=FakeClassifierModel())


def test_rejects_unknown_default_tier():
    with pytest.raises(ValueError, match="default_tier"):
        ModelRouterMiddleware(
            tiers=make_tiers(),
            classifier_model=FakeClassifierModel(),
            default_tier="does-not-exist",
        )


def test_defaults_to_first_tier_when_default_tier_omitted():
    router = ModelRouterMiddleware(tiers=make_tiers(), classifier_model=FakeClassifierModel(error=RuntimeError()))
    state = {"messages": [HumanMessage(content="whatever")]}

    assert router.before_model(state, runtime=None) == {"selected_tier": "fast"}


@pytest.mark.asyncio
async def test_abefore_model_classifies_and_caches():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(tier="performance"),
        default_tier="balanced",
    )
    state = {"messages": [HumanMessage(content="rewrite the whole pipeline")]}

    update = await router.abefore_model(state, runtime=None)
    assert update == {"selected_tier": "performance"}

    state["selected_tier"] = "performance"
    assert await router.abefore_model(state, runtime=None) is None


@pytest.mark.asyncio
async def test_abefore_model_falls_back_on_error():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(error=RuntimeError("boom")),
        default_tier="balanced",
    )
    state = {"messages": [HumanMessage(content="fix the flaky test")]}

    assert await router.abefore_model(state, runtime=None) == {"selected_tier": "balanced"}


@pytest.mark.asyncio
async def test_awrap_model_call_overrides_model_for_selected_tier():
    router = ModelRouterMiddleware(
        tiers=make_tiers(),
        classifier_model=FakeClassifierModel(tier="fast"),
        default_tier="balanced",
    )
    request = FakeModelRequest(state={"selected_tier": "performance"})

    async def handler(req):
        return req.model

    result = await router.awrap_model_call(request, handler=handler)

    assert result == "performance-model"
