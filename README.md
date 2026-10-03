# model-router

A LangChain agent [middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
that routes each conversation thread to the cheapest model tier that can
handle it, instead of sending every request to your most capable (and most
expensive) model.

Inspired by LangChain's [_How to Build a Model Router in the
Harness_](https://x.com/sydneyrunkle/status/2105704739585565066), which
describes cutting Open SWE's median cost per thread by 64% with no
measurable quality regression by routing requests to one of three model
tiers based on task complexity. This project is a small, reusable,
provider-agnostic implementation of that idea as a drop-in middleware for
`langchain.agents.create_agent`.

## How it works

1. **You define tiers.** Each `ModelTier` pairs a model (any LangChain
   chat model, from any provider) with a short, plain-language description
   of the kind of task that belongs on it.
2. **On a thread's first message**, `ModelRouterMiddleware` asks a
   classifier model to pick the cheapest tier likely to complete the task,
   using your tier criteria as the rubric.
3. **The choice is cached** on agent state (`selected_tier`) and reused for
   every later turn in that thread — classification runs once per thread,
   not once per call.
4. **If classification fails or returns something unexpected**, the router
   falls back to a `default_tier` you configure, so a flaky classifier
   call never breaks the agent.

The routing decision lives in the harness (as middleware), not behind a
generic model gateway, because picking the right tier needs the same
task-specific context (what this agent is for, what "simple" vs. "hard"
means for its tasks) that the harness already has and a gateway typically
doesn't.

## Install

Not on PyPI yet — install straight from GitHub:

```bash
pip install git+https://github.com/whyevenquestion1t/model-router.git
```

or clone it and install in editable mode:

```bash
git clone https://github.com/whyevenquestion1t/model-router.git
cd model-router
pip install -e .
```

## Usage

```python
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
        criteria="Multi-file refactors, tricky debugging, ambiguous requests.",
    ),
]

router = ModelRouterMiddleware(
    tiers=tiers,
    classifier_model=init_chat_model("claude-haiku-4-5"),
    default_tier="balanced",  # used if classification fails
)

agent = create_agent(
    model="claude-sonnet-4-6",  # overridden per-thread by the router
    middleware=[router],
    tools=[...],
)
```

See [`examples/basic_usage.py`](examples/basic_usage.py) for a runnable
version.

Tiers aren't tied to a single provider — mix and match freely, e.g. an
open-weight model for `fast` and two different closed providers for
`balanced`/`performance`, exactly as LangChain did in their writeup.

## Writing good tier criteria

The classifier is only as good as the criteria you give it. Base them on:

- **Your own traffic.** Look at what your agent is actually asked to do
  (task type, typical cost, typical turn count) before deciding where the
  lines between tiers go.
- **What each model is actually good at**, per its provider's own
  guidance — don't rely on a single generic benchmark score.

Write each tier's `criteria` the way you'd brief a teammate on what kind of
work to hand them, not as a one-word label.

## Measuring whether it's working

A router only helps if quality doesn't regress. Before trusting it in
production:

- Track an outcome metric you already have (task success rate, user
  thumbs up/down, PR merge rate, etc.) split by routed vs. not routed.
- A/B test against a single-model baseline on live traffic if you don't
  have a representative offline eval set — it's usually easier to get a
  trustworthy signal that way than building a graded offline dataset.

## Async

`ModelRouterMiddleware` implements both the sync (`before_model`,
`wrap_model_call`) and async (`abefore_model`, `awrap_model_call`) hooks, so
it works the same with `agent.invoke(...)` and `agent.ainvoke(...)`.

## Limitations / what this doesn't do

- **Routes once per thread, not mid-thread.** If a thread's topic or
  difficulty shifts partway through, the original tier choice sticks. Swapping
  models mid-thread also throws away any provider-side prompt cache for that
  thread.
- **Doesn't route subagents independently** — if your agent spawns
  subagents, they pick their own models unless you wire this middleware
  into them too.
- **Classification adds one extra model call** at the start of each
  thread. Use a small, cheap model for `classifier_model` to keep that
  overhead negligible relative to the savings.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
