"""Wire ModelRouterMiddleware into a `create_agent` agent.

Requires API keys for whichever providers you point the tiers and the
classifier at (this example uses Anthropic models for all of them, but
tiers can mix providers freely).
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from model_router import ModelRouterMiddleware, ModelTier

tiers = [
    ModelTier(
        name="fast",
        model=init_chat_model("claude-haiku-4-5"),
        criteria="Trivial lookups, formatting, simple Q&A, no-op/test runs.",
    ),
    ModelTier(
        name="balanced",
        model=init_chat_model("claude-sonnet-4-6"),
        criteria="Everyday feature work and bug fixes in a familiar codebase.",
    ),
    ModelTier(
        name="performance",
        model=init_chat_model("claude-opus-4-6"),
        criteria="Multi-file refactors, tricky debugging, unfamiliar or ambiguous requests.",
    ),
]

router = ModelRouterMiddleware(
    tiers=tiers,
    classifier_model=init_chat_model("claude-haiku-4-5"),
    default_tier="balanced",
)

agent = create_agent(
    model="claude-sonnet-4-6",  # overridden per-thread by the router
    middleware=[router],
    tools=[],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "Rename `foo` to `bar` in utils.py"}]})
    print(result["messages"][-1].content)
