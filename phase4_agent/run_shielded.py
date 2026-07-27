"""
Runs the SHIELDED agent — generates a candidate action, checks it against
the process graph, and asks the model to reconsider if it violates conformance.
Section 8.5: this produces the headline shielded-vs-baseline comparison.
"""
import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import agent_step, parse_action
from phase4_agent.shielding.graph_verifier import shield_score, load_process_graph

MAX_STEPS = 15
MAX_SHIELD_RETRIES = 2


def run_task_shielded(task_id: str = "browsergym/miniwob.click-test", headless: bool = True):
    env = gym.make(task_id, headless=headless)
    obs, info = env.reset()
    goal = obs.get("goal_object", [{"text": "unknown"}])[0].get("text", "unknown")

    process_graph = load_process_graph()
    action_history = []

    for step in range(MAX_STEPS):
        action = None
        for retry in range(MAX_SHIELD_RETRIES + 1):
            raw_action = agent_step(obs, goal)
            candidate = parse_action(raw_action)

            # Build the candidate sequence: history + this new action
            candidate_sequence = action_history + [candidate]
            cost = shield_score(candidate_sequence, process_graph)

            if cost == 0 or retry == MAX_SHIELD_RETRIES:
                action = candidate
                if cost > 0:
                    print(f"Step {step}: shield flagged violation (cost={cost}), proceeding anyway after {retry} retries")
                break
            else:
                print(f"Step {step}: shield blocked '{candidate}' (cost={cost}), retrying ({retry + 1}/{MAX_SHIELD_RETRIES})")

        print(f"Step {step}: {action}")
        action_history.append(action)

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
    result = run_task_shielded()
    print(f"✅ Shielded agent run complete. Final reward: {result}")
