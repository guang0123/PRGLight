from pipeline import Pipeline
import os
import time
from multiprocessing import Process
import argparse

from config import get_config

multi_process = True


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_set', type=str, default='light')
    parser.add_argument('--roadnet_file', type=str, default='roadnet_light.json')
    parser.add_argument('--vehicle_file', type=str, default='anon_light.json')
    parser.add_argument('--GNN_W_file', type=str, default='GNN_dataset/W.csv')
    parser.add_argument('--GNN_V_file', type=str, default='GNN_dataset/V.csv')
    parser.add_argument('--num_col', type=int, default=1)
    parser.add_argument('--num_row', type=int, default=1)
    return parser.parse_args()

def pipeline_wrapper(dic_exp_conf, dic_agent_conf, dic_traffic_env_conf, dic_path):
    ppl = Pipeline(dic_exp_conf=dic_exp_conf,  # experiment config
                   dic_agent_conf=dic_agent_conf,  # RL agent config
                   dic_traffic_env_conf=dic_traffic_env_conf,  # the simolation configuration
                   dic_path=dic_path  # where should I save the logs?
                   )
    ppl.run(multi_process=multi_process)

    return

def main(data_set,roadnet_file,vehicle_file,GNN_W_file,GNN_V_file,num_col,num_row): 
    dic_agent_conf, dic_exp_conf, dic_traffic_env_conf = get_config()

    current_time = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    dic_path = {
        'PATH_TO_MODEL': os.path.join('model',str(current_time)),
        'PATH_TO_WORK_DIRECTORY': os.path.join('records',str(current_time)),
        'PATH_TO_DATA': os.path.join('data', data_set),
        'PATH_TO_ERROR': os.path.join('errors',str(current_time)),
        'PATH_TO_GNN_W': GNN_W_file,
        'PATH_TO_GNN_V': GNN_V_file,
        'PATH_TO_PRETRAIN_MODEL': os.path.join('model','initial'),
        'PATH_TO_PRETRAIN_WORK_DIRECTORY': os.path.join('records','initial'),
    }

    dic_agent_conf['TRAFFIC_FILE'] = vehicle_file
    dic_exp_conf['TRAFFIC_FILE'] = [vehicle_file]
    dic_exp_conf['ROADNET_FILE'] = roadnet_file

    dic_traffic_env_conf['NUM_INTERSECTIONS'] = num_row*num_col
    dic_traffic_env_conf['NUM_ROW'] = num_row
    dic_traffic_env_conf['NUM_COL'] = num_col
    dic_traffic_env_conf['TRAFFIC_FILE'] = vehicle_file
    dic_traffic_env_conf['ROADNET_FILE'] = roadnet_file

    dic_exp_conf["NUM_ROUNDS"] = 30  # total round
    dic_agent_conf["LEARNING_RATE"] = 0.0005  # learning step
    dic_agent_conf["BATCH_SIZE"] = 32
    dic_exp_conf["RUN_COUNTS"] = 3600

    if multi_process:
        ppl = Process(target=pipeline_wrapper,
                      args=(dic_exp_conf,
                            dic_agent_conf,
                            dic_traffic_env_conf,
                            dic_path))
        ppl.start()
        ppl.join()
    else:
        pipeline_wrapper(dic_exp_conf=dic_exp_conf,
                        dic_agent_conf=dic_agent_conf,
                        dic_traffic_env_conf=dic_traffic_env_conf,
                        dic_path=dic_path)


if __name__ == "__main__":
    args = parse_args()
    main(args.data_set,args.roadnet_file,args.vehicle_file,args.GNN_W_file,args.GNN_V_file,args.num_col,args.num_row)





