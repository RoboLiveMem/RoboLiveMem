"""RoboLiveMem: is robot memory solved when the world does not wait?

    robolivemem.protocol  the environment protocol (server side): observations, commands, the clock
    robolivemem.env       client: one interface for every tier (connect, reset, act)
    robolivemem.agents    Agent base class, the reference chat harness, external harnesses
    robolivemem.eval      run an agent on the paper's episodes of a tier (python -m robolivemem.eval)
    robolivemem.report    per-task and per-tier tables (python -m robolivemem.report)
    robolivemem.mcp       the protocol as MCP tools for tool-using harnesses
    robolivemem.clock     paused, live and fixed clocks (Eq. 1)
    robolivemem.metrics   live drop, execution and memory terms with paired bootstrap intervals (Eq. 2-4)
    robolivemem.commit    the commit check (Eq. 3)
    robolivemem.models    one chat interface for OpenAI, Anthropic, Gemini, Bedrock and OpenAI-compatible servers
    robolivemem.tabletop  a tiny 2-D version of the Grandpa's mug family (python -m robolivemem; server: .serve)
"""
__version__ = "0.2.0"

from .clock import Clock, PAUSED, LIVE, fixed  # noqa: F401
