"""The orchestration feature: run_plan and start_room."""
from deep_agents_demo.features.orchestration.plans import PLAN_PROMPT, run_plan
from deep_agents_demo.features.orchestration.rooms import ROOM_PROMPT, start_room


def orchestration_feature():
    from deep_agents_demo.core.feature import Feature  # agent.py imports this module
    return Feature(name="orchestration", prompt=PLAN_PROMPT + ROOM_PROMPT,
                   tools=(run_plan, start_room))
