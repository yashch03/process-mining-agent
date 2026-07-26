"""
Runs the baseline (unshielded) agent on a MiniWoB++ task end to end.
"""
import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import agent_step, parse_action

MAX_STEPS = 15


def run_task(task_id: str = "browsergym/miniwob.click-test", headless: bool = True):
    env = gym.make(task_id, headless=headless)
    obs, info = env.reset()
    goal = obs.get("goal_object", [{"text": "unknown"}])[0].get("text", "unknown")

    for step in range(MAX_STEPS):
        raw_action = agent_step(obs, goal)
        action = parse_action(raw_action)
        print(f"Step {step}: {action}")

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception as e:
            print(f"Action failed: {e}")
            break

        if terminated or truncated:
            print(f"Task ended. Reward: {reward}")
            env.close()
            return reward

    env.close()
    print("Max steps reached without completion")
    return 0


if __name__ == "__main__":
    result = run_task()
    print(f"✅ Baseline agent run complete. Final reward: {result}")
