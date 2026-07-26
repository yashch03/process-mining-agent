"""
NIM-backed agent: takes a BrowserGym observation, asks GLM-5.2 for an action.
This is Section 8.3's baseline unshielded agent.
"""
import os
import re
import time
import openai
from browsergym.utils.obs import flatten_axtree_to_str

client = openai.OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
)

SYSTEM_PROMPT = """You are a web agent completing a browsing task.
You will be given the task goal and the current page's accessibility tree.
Respond with EXACTLY ONE action in this format:
click(bid="123")   OR   fill(bid="123", value="text")   OR   send_msg_to_user("done")
No explanation, just the action."""


def agent_step(observation: dict, task_goal: str, model: str = "z-ai/glm-5.2", max_retries: int = 4) -> str:
    axtree_obj = observation.get("axtree_object")
    axtree = flatten_axtree_to_str(axtree_obj)[:4000] if axtree_obj else "(no page content available)"

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Goal: {task_goal}\n\nPage:\n{axtree}"},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"Rate limited, waiting {wait}s before retry {attempt + 1}/{max_retries}...")
            time.sleep(wait)
    raise RuntimeError("Exceeded max retries due to rate limiting")


def parse_action(action_str: str) -> str:
    match = re.search(r'(click|fill|send_msg_to_user)\([^)]*\)', action_str)
    return match.group(0) if match else action_str
