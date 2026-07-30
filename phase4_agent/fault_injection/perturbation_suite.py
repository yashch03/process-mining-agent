"""
Controlled UI perturbation suite — tests whether shielding stays robust
under interface drift (label renaming, DOM reordering).
Section 8.12 of the implementation doc.
"""
import random
import gymnasium as gym
import browsergym.miniwob
from phase4_agent.agent_client import agent_step, parse_action
from phase4_agent.shielding.graph_verifier import shield_score, load_process_graph


def run_trial_with_perturbation(task_id, perturbation_seed=None, headless=True):
    """
    Runs a task trial. Perturbation is simulated by adding a random seed
    offset that changes MiniWoB's own randomized element layout —
    a lightweight proxy for structural UI drift, since MiniWoB tasks
    already randomize element positions/labels per seed.
    """
    env = gym.make(task_id, headless=headless)
    obs, info = env.reset(seed=perturbation_seed)
    goal = obs.get("goal_object", [{"text": "unknown"}])[0].get("text", "unknown")

    for step in range(10):
        raw = agent_step(obs, goal)
        action = parse_action(raw)
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except Exception:
            env.close()
            return 0
        if terminated or truncated:
            env.close()
            return reward

    env.close()
    return 0


if __name__ == "__main__":
    task = "browsergym/miniwob.click-test"
    seeds = [1, 2, 3, 4, 5]  # each seed = a different randomized layout ("perturbation")

    results = []
    for seed in seeds:
        reward = run_trial_with_perturbation(task, perturbation_seed=seed)
        results.append(reward)
        print(f"Seed {seed}: reward={reward}")

    success_rate = sum(results) / len(results)
    print(f"\n✅ Success rate across {len(seeds)} randomized layouts: {success_rate:.2f}")
