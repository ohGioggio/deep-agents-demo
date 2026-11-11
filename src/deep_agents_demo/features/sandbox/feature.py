"""What the agent is told about working in a sandbox."""
from deep_agents_demo.core.feature import Feature


def sandbox_feature(workdir, openshell_skill=False) -> Feature:
    """Rules for working in a sandbox; with OpenShell, also its skill folder."""
    prompt = (f"You are using a sandbox. Keep code and output under {workdir}; when a skill or "
              f"the user names a path starting with /, put it under {workdir} instead. "
              "Local /instructions paths contain skills and the images the user attached "
              "(uploads/); create your own files under the workdir. ")
    if openshell_skill:
        # The sandbox skill describes the OpenShell image; offering it without that
        # sandbox misleads the agent (it told users to pip install inside /sandbox).
        prompt += ("Before your first shell command (execute), read the sandbox skill and "
                   "follow it for installing packages and handling blocked network access; "
                   "other tools don't need it. ")
    # Asking was not enough: the agent skipped the skill and tried pip, apt and
    # get-pip before finding uv. So execute is refused until the skill is read.
    return Feature(name="sandbox", prompt=prompt,
                   skill_dirs=("sandbox-skills/",) if openshell_skill else (),
                   needs_skill=(("execute", "sandbox"),) if openshell_skill else ())
