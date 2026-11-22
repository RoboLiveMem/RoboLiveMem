"""RoboLiveMem: is robot memory solved when the world does not wait?

    robolivemem.clock     paused, live and fixed clocks (Eq. 1)
    robolivemem.metrics   live drop, execution and memory terms with paired bootstrap intervals (Eq. 2-4)
    robolivemem.commit    the commit check (Eq. 3)
    robolivemem.models    one chat interface for OpenAI, Anthropic, Gemini, Bedrock and OpenAI-compatible servers
    robolivemem.tabletop  a tiny 2-D version of the Grandpa's mug family to try everything in seconds
"""
__version__ = "0.1.0"

from .clock import Clock, PAUSED, LIVE, fixed  # noqa: F401
