"""
Confirms BrowserGym can actually launch a browser and step through
one MiniWoB++ task. No agent intelligence yet — just infrastructure check.
"""
import gymnasium as gym
import browsergym.miniwob

env = gym.make("browsergym/miniwob.click-test", headless=True)
obs, info = env.reset()

print("Observation keys:", list(obs.keys()))
print("Goal:", obs.get("goal_object", "N/A"))

env.close()
print("✅ BrowserGym environment launches successfully")
