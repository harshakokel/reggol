# Reggol

Reggol (logger in reverse) is a logging util for ML/AI experiments. It is compatible with viskit for visualizing the results.


### Quick Start

#### Setup 

To use Reggol effectively put all the experiment details: hyper parameters, algorithm, and domain name in a dictionary and pass it to `setup_logger` as shown below. Reggol will save this dictionary as json in the experiment folder along with the git commit id and patch file.  

```python 
from reggol import setup_logger
import os

variant = {'exp_prefix': "custom_experminet",
           'algorithm': "baseline",
           'domain': "gridworld",
           'batch_size': 5,
           'learning_rate': 0.02}

logger = setup_logger(variant['exp_prefix'], variant, exp_id=os.getpid())
```

Reggol can record logs either as print statements in a text file `debug.log` (c.f. `text_log_file` in `setup_logger` to customize this file name) or as a tabular data in `progress.csv` file (c.f. `tabular_log_file` in `setup_logger` to customize)

#### Text log 

To log a string or sentence use `logger` as below, this will print the log in stdout as well as write it to the `debug.log` file.

```python
itr = 0
logger.log(f"Starting iteration {itr}")
```

#### Tabular log 

To record experiment results in each iteration to `progress.csv` reggol provides two option, either log each value individually using `record_tabular` or record multiple values using `record_dict`. When the iteration is over, dump all the values to csv file (as wel to stdout) using `dump_tabular`. 

```python
for i in range(5):
    logger.record_tabular("mse",0.0001*i)
    logger.record_dict({'iteration':i,
                        'sample size': 10*i)
    logger.dump_tabular()
```

For additional reference, see [sample usage](./sampe_usage.py). 

### Integration with third party

1. [Stable Baselines3](https://stable-baselines3.readthedocs.io) 

```python
# Setup Logger
from reggol import setup_logger
logger = setup_logger(variant['exp_prefix'], variant=variant)

# Configure the SB3 logger
from stable_baselines3.common.logger import configure
new_logger = configure(logger.get_snapshot_dir(), ["stdout", "csv"])

# Use the logger
from stable_baselines3 import DQN
model = DQN(...)
model.set_logger(new_logger)
```

2. [ACME RL](https://github.com/deepmind/acme)

```python

# Setup Logger
from reggol import setup_logger
logger = setup_logger(variant['exp_prefix'], variant=variant)

# Configure the ACME agent and environment
from acme.agents.jax import ppo
import gym
agent = ppo(...)
environment = gym.make(...)

# Use the logger
train_loop = acme.EnvironmentLoop(environment, agent, logger)
```

### Credits

Code it partially taken from following sources

* https://github.com/vitchyr/rlkit 

