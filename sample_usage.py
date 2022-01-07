from reggol import setup_logger
import os
import time

variant = {'exp_prefix': "custom_experiment",
           'algorithm': "baseline",
           'domain': "gridworld",
           'batch_size': 5,
           'learning_rate': 0.02}

logger = setup_logger(variant['exp_prefix'],
                      variant,
                      exp_id=os.getpid(),
                      tabular_log_file='record.csv',
                      text_log_file='output.log',
                      variant_log_file='dict.json',
                      base_log_dir='./sample_usage_log/',
                      snapshot_mode='last')

logger.log("Starting Experiment")

x = 0
for i in range(5):
    logger.log(f"Started iteration {i}")
    s1 = time.process_time()
    time.sleep(5)
    s2 = time.process_time()
    logger.record_tabular("mse", 0.0001 * i)
    logger.record_dict({'iteration': i,
                        'sample size': 10 * i,
                        'training time': s2 - s1})
    sample_snapshot = dict(firstname="Harsha",
                           lastname="Kokel",
                           spouse=dict(firstname="Abhinav",
                                       lastname="Prakash"),
                           x=x)
    logger.save_itr_params(i, sample_snapshot, mode='json')
    logger.save_itr_params(i, sample_snapshot, mode='pickle')
    logger.dump_tabular()
    x += 1

logger.save_extra_data(variant, file_name='extra_data.pkl', mode='pickle')
logger.log("Ending Experiment")
