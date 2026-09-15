"""Run a few episodes with randomly selected prices."""

from __future__ import annotations

from environment.dynamic_pricing_env import DynamicPricingEnv


def run_episode(env: DynamicPricingEnv, episode_id: int) -> float:
    observation, info = env.reset()
    total_revenue = 0.0
    step = 0
    terminated = False

    print(f"\n=== Episode {episode_id} ===")
    print(
        f"Start | tickets={info['tickets_remaining']} "
        f"days={info['days_remaining']} obs={observation}"
    )

    while not terminated:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_revenue += info["revenue"]
        step += 1
        print(
            f"Day step={step:02d} | price=${info['price']:.0f} "
            f"demand={info['demand']} sold={info['tickets_sold']} "
            f"revenue=${info['revenue']:.0f} "
            f"left={info['tickets_remaining']} "
            f"days_left={info['days_remaining']} "
            f"reward={reward:.1f}"
        )
        assert truncated is False

    sell_through = (
        (env.initial_inventory - info["tickets_remaining"]) / env.initial_inventory
    )
    print(
        f"Done  | total_revenue=${total_revenue:.0f} "
        f"sell_through={sell_through:.1%} "
        f"unsold={info['tickets_remaining']}"
    )
    return total_revenue


def main() -> None:
    env = DynamicPricingEnv(
        initial_inventory=100,
        selling_days=20,
        demand_level=1.0,
        terminal_inventory_penalty=0.0,
        seed=42,
    )

    revenues = []
    for episode_id in range(1, 4):
        revenues.append(run_episode(env, episode_id))

    print("\n=== Summary ===")
    print(f"Episode revenues: {[round(r, 1) for r in revenues]}")
    print(f"Average revenue:  ${sum(revenues) / len(revenues):.1f}")


if __name__ == "__main__":
    main()
