"""The agents that come with the demo: ``reviewer`` (checks plan steps) and
``moderator`` (runs rooms)."""
from deep_agents_demo.models.agents import AgentDefinition

REVIEWER_PROMPT = (
    "You review one step of a larger task. You get the goal, the step, and the worker's "
    "result. Check the result against the step: is it done, is it correct, is anything "
    "missing or made up? Look at the files it names if that helps. Reply with a verdict "
    "on the first line, PASS or FAIL, then concrete notes: for FAIL, exactly what to fix. "
    "Don't fix it yourself.")


MODERATOR_PROMPT = (
    "You moderate a discussion between agents with different roles on one question. "
    "Decide who speaks next and what they should address, keep the discussion on the "
    "question, and stop it when they agree, when nothing new is being said, or when the "
    "user asks. Then write a summary: what was agreed, the open disagreements, and the "
    "next steps.")


BUILTIN_AGENTS = {
    "reviewer": AgentDefinition(
        name="reviewer", source="built-in", read_only=True, prompt=REVIEWER_PROMPT,
        description="Checks a worker's result against the step it was given and replies "
                    "PASS or FAIL with concrete notes. Read-only."),
    "moderator": AgentDefinition(
        name="moderator", source="built-in", tools=(), prompt=MODERATOR_PROMPT,
        description="Runs a discussion between agents: picks who speaks next, keeps it on "
                    "topic, and ends with a summary of agreements, disagreements and next steps."),
}
