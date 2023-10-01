from stable_baselines3 import PPO

from env.dummy_env import DummyEnv


def train():
    model = PPO("MlpPolicy", DummyEnv()).learn(10_000)
    return model


if __name__ == "__main__":
    train()
