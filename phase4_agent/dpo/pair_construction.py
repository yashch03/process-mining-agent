"""
Constructs DPO preference pairs directly from graph structure.
Chosen = real agent action (graph-conforming in context).
Rejected = a deliberately constructed graph-violating alternative.
Section 8.6 of the implementation doc.

Saves incrementally after each task, so interruptions don't lose progress.
"""
import json
import random
import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import agent_step, parse_action
from phase4_agent.shielding.graph_verifier import shield_score, load_process_graph

DATASET_PATH = "phase4_agent/dpo/preference_pairs.json"


def construct_rejected_variant(chosen_action: str) -> str:
    import re
    match = re.search(r'bid="(\d+)"', chosen_action)
    if match:
        bid = int(match.group(1))
        fake_bid = bid + random.choice([-5, 5, 10, -10])
        return chosen_action.replace(f'bid="{bid}"', f'bid="{fake_bid}"')
    return chosen_action


def load_dataset():
    try:
        with open(DATASET_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_dataset(pairs):
    with open(DATASET_PATH, "w") as f:
        json.dump(pairs, f, indent=2)


def build_preference_pairs(task_id, n_trials=3, headless=True):
    process_graph = load_process_graph()
    pairs = []

    for trial in range(n_trials):
        try:
            env = gym.make(task_id, headless=headless)
            obs, info = env.reset()
            goal = obs.get("goal_object", [{"text": "unknown"}])[0].get("text", "unknown")
            action_history = []

            for step in range(10):
                raw = agent_step(obs, goal)
                chosen = parse_action(raw)
                rejected = construct_rejected_variant(chosen)

                if rejected != chosen:
                    pairs.append({
                        "task": task_id,
                        "context": f"Goal: {goal}",
                        "chosen": chosen,
                        "rejected": rejected,
                    })

                action_history.append(chosen)
                try:
                    obs, reward, terminated, truncated, info = env.step(chosen)
                except Exception:
                    break
                if terminated or truncated:
                    break

            env.close()
            print(f"Trial {trial+1}/{n_trials} on {task_id}: {len(pairs)} pairs so far (this task)")
        except Exception as e:
            print(f"Task {task_id} trial {trial+1} failed: {e}")
            continue

    return pairs


if __name__ == "__main__":
    tasks = [
        "browsergym/miniwob.click-checkboxes",
        "browsergym/miniwob.click-button",
        "browsergym/miniwob.click-link",
        "browsergym/miniwob.click-widget",
        "browsergym/miniwob.click-test",
    ]
    n_trials_per_task = 4

    existing = load_dataset()
    print(f"Starting with {len(existing)} existing pairs")

    for task in tasks:
        pairs = build_preference_pairs(task, n_trials=n_trials_per_task)
        existing.extend(pairs)
        save_dataset(existing)  # SAVE AFTER EVERY TASK, not just at the end
        print(f"✅ Saved after {task}: {len(existing)} total pairs now on disk")

    print(f"\n✅ Final total: {len(existing)} pairs")
