import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import axtree_to_text

env = gym.make("browsergym/miniwob.click-test", headless=True)
obs, info = env.reset()

print("=== GOAL ===")
print(obs.get("goal_object"))
print("\n=== AXTREE (first 2000 chars) ===")
print(axtree_to_text(obs.get("axtree_object"))[:2000])

env.close()
