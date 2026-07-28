"""
Constructs DPO preference pairs directly from graph structure.
Chosen = real agent action (graph-conforming in context).
Rejected = a deliberately constructed graph-violating alternative.
Section 8.6 of the implementation doc.
"""
import json
import random
import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import agent_step, parse_action
from phase4_agent.shielding.graph_verifier import shield_score, load_process_graph


def construct_rejected_variant(chosen_action: str) -> str:
    """
    Given a real chosen action like click(bid="30"), construct a plausible
    but likely-invalid variant by perturbing the bid to a nearby number —
    simulating a hallucinated/incorrect element reference.
    """
    import re
    match = re.search(r'bid="(\d+)"', chosen_action)
    if match:
        bid = int(match.group(1))
        fake_bid = bid + random.choice([-5, 5, 10, -10])
        return chosen_action.replace(f'bid="{bid}"', f'bid="{fake_bid}"')
    return chosen_action  # fallback, unchanged


def build_preference_pairs(task_id, n_trials=3, headless=True):
    process_graph = load_process_graph()
    pairs = []

    for trial in range(n_trials):
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
        print(f"Trial {trial+1}/{n_trials} on {task_id}: {len(pairs)} pairs so far")

    return pairs


if __name__ == "__main__":
    all_pairs = []
    tasks = ["browsergym/miniwob.click-checkboxes", "browsergym/miniwob.click-button"]

    for task in tasks:
        pairs = build_preference_pairs(task, n_trials=3)
        all_pairs.extend(pairs)

    with open("phase4_agent/dpo/preference_pairs.json", "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\n✅ Generated {len(all_pairs)} preference pairs")
    print(f"Saved to phase4_agent/dpo/preference_pairs.json")
