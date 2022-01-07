# Reggol

Reggol (spelling "logger" backwards; in the spirit of UT Dallas) is a logging util for ML/AI experiments. It is compatible with [viskit](https://github.com/harshakokel/viskit) for visualizing and monitoring the results while experiments are running. That's right, **while experiments are running**.

**Why Reggol?** In my experience, we need text logs to monitor experiments while development, but need tabular (or csv) files to plot and compare the various experiments. Reggol helps achieve both in a seamless manner, while also keeping track of all the hyperparameters used in each experiment for easy comparison. So go Reggol.

### Usage guide

#### Setup Logger 

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

Reggol can record logs either as print statements in a text file `debug.log` (c.f. `text_log_file` in `setup_logger` to customize this file name) or as a tabular data in `progress.csv` file (c.f. `tabular_log_file` in `setup_logger` to customize). Reggol output for the above setup is provided in [sample_usage_log dir](./sample_usage_log/).

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

For additional reference, see the [python file `sample_usage.py`](./sample_usage.py) and its output in the [folder `sample_usage_log`](./sample_usage_log). 

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

### Installation

To use reggol use following commands

```
git clone https://github.com/harshakokel/reggol.git
cd reggol
pip install -e .
```


### Credits

Code it partially taken from following sources

* https://github.com/vitchyr/rlkit 

