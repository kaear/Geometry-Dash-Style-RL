"""Verify Gymnasium API and run random + simple heuristic baselines."""
from stable_baselines3.common.env_checker import check_env
from geometry_env import GeometryDashEnv


def play(env, mode, episodes=20):
    successes, progresses = 0, []
    for ep in range(episodes):
        observation, _ = env.reset(seed=1000 + ep)
        done = False
        while not done:
            if mode == "random":
                action = env.action_space.sample()
            else:
                # Starter heuristic: jump shortly before the next spike.
                grounded = observation[2] > 0.5
                distance = observation[3] * 25.0
                action = int(grounded and 1.5 < distance < 2.7)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        successes += int(info["success"])
        progresses.append(info["progress"])
    print(f"{mode}: success={successes}/{episodes}; mean progress={sum(progresses)/episodes:.1%}")


if __name__ == "__main__":
    env = GeometryDashEnv()
    check_env(env, warn=True)
    play(env, "random")
    play(env, "heuristic")
