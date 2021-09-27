import ray
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from itertools import count
from .env import JetFighter


ray.init()


config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 1
config["num_workers"] = 6
config["env_config"] = {}
config["model"] = {
    "conv_filters": [[3, [2, 2], 2], [16, [8, 8], 4], [32, [4, 4], 2], [48, [12, 15], 1]],
    "post_fcnet_hiddens": [48, 32],
}
config["framework"] = "torch"
trainer = ppo.PPOTrainer(env=JetFighter, config=config)
# Can optionally call trainer.restore(path) to load a checkpoint.

for i in count():
    # Perform one iteration of training the policy with PPO
    result = trainer.train()
    print(pretty_print(result))

    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
