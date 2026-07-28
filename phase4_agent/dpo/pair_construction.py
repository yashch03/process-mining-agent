"""
Constructs DPO preference pairs from shielded agent runs.
Chosen = graph-conforming action, Rejected = graph-blocked candidate.
Section 8.6 of the implementation doc.
"""
import json
import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import agent_step, parse_action
from phase4_agent.shielding.graph_verifier import shield_score, load_process_graph


def generate_candidates(observation, task_goal, n_candidates=3):
    """Generate multiple diverse candidate actions using varied temperature."""
    candidates = []
    temperatures = [0.3, 0.7, 1.0]
    for i in range(n_candidates):
        temp = temperatures[i % len(temperatures)]
        raw = agent_step(observation, task_goal, temperature=temp)
        candidates.append(parse_action(raw))
    return list(set(candidates))


def build_preference_pairs(task_id, n_trials=3, headless=True):
    process_graph = load_process_graph()
    pairs = []

    for trial in range(n_trials):
        env = gym.make(task_id, headless=headless)
        obs, info = env.reset()
        goal = obs.get("goal_object", [{"text": "unknown"}])[0].get("text", "unknown")
        action_history = []

        for step in range(10):
            candidates = generate_candidates(obs, goal)
            scored = [(c, shield_score(action_history + [c], process_graph)) for c in candidates]

            conforming = [c for c, cost in scored if cost == 0]
            blocked = [c for c, cost in scored if cost > 0]

            for c in conforming:
                for b in blocked:
                    pairs.append({
                        "task": task_id,
                        "context": f"Goal: {goal}",
                        "chosen": c,
                        "rejected": b,
                    })

            action = conforming[0] if conforming else candidates[0]
            action_history.append(action)

            try:
                obs, reward, terminated, truncated, info = env.step(action)
            except Exception:
                break
            if terminated or truncated:
                break

        env.close()
        print(f"Trial {trial+1}/{n_trials} on {task_id}: {len(pairs)} pairs so far")

    return pairs


if __name__ == "__main__":
    all_pairs = []
    tasks = ["browsergym/miniwob.click-checkboxes"]  # the task that actually showed shield activity

    for task in tasks:
        pairs = build_preference_pairs(task, n_trials=2)
        all_pairs.extend(pairs)

    with open("phase4_agent/dpo/preference_pairs.json", "w") as f:
        json.dump(all_pairs, f, indent=2)

    print(f"\n✅ Generated {len(all_pairs)} preference pairs")
    print(f"Saved to phase4_agent/dpo/preference_pairs.json")
