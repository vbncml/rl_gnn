import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--actor_model', dest='actor_model', type=str, default='')
    parser.add_argument('--critic_model', dest='critic_model', type=str, default='')

    args = parser.parse_args()

    return args