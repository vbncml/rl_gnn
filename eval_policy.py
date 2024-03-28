from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib import animation
from network import GCN
import sys

def _save_frames_as_gif(frames, path='./', filename='dog2_env-recurrent_gcn_k_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=20)


def _log_summary(ep_len, ep_reward, episode):
    ep_len = str(round(ep_len, 2))
    ep_reward = str(round(ep_reward, 2))

    print(flush=True)
    print(f"-------------------- Episode #{episode} --------------------", flush=True)
    print(f"Episodic Length: {ep_len}", flush=True)
    print(f"Episodic Return: {ep_reward}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)

def rollout(policy, env, render):

    while True:
        obs, _ = env.reset()
        done = False

        t=0
        edge_index = env.get_edge_index()
        ep_len = 0
        ep_reward = 0
        frames = [] 
        
        while not done:
            
            t += 1

            if render:
                frames.append(env.render())
                env.render()
            
            
            # action = policy(obs).detach().numpy()
            # else:
            # action = policy(obs, edge_index).detach().numpy().reshape(17,)
            action = policy(obs, edge_index).detach().numpy()
            obs, reward, done, _, _ = env.step(action)

            ep_reward += reward 
        
            #if t > 200 or done==True:
            # _save_frames_as_gif(frames)
            # sys.exit(0)
        
        
        
        yield ep_len, ep_reward, frames
    
    # _save_frames_as_gif(frames)

def eval_policy(policy, env, render=True):

    for episode, (ep_len, ep_reward, frames) in enumerate(rollout(policy, env, render)):
        _log_summary(ep_len=ep_len, ep_reward=ep_reward, episode=episode)
        # if ep_reward > -10:
            #  _save_frames_as_gif(frames)
            #  sys.exit(0)