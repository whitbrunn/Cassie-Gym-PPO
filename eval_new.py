import torch
import gym
import gym_pkg  # 确保 gym_pkg 已正确安装并包含所需的环境
import numpy as np
import time


# 加载训练好的模型
def load_model(model_path, device):
    model = torch.load(model_path)  # Replace "model.pt" with the path to your .pt file

    model.eval()  # 设置模型为评估模式
    
    model.to(device)
    return model

# 运行环境并进行可视化
def visualize_policy(env, model, device, render=True):
    state, _ = env.reset()  # 初始化环境
    done = False
    total_reward = 0.0

    while not done:
        if render:
            env.render()  # 可视化环境
            time.sleep(0.02)  # 控制可视化帧率

        # 状态转换为张量并传入模型
        
        state_array = np.array(state)  # 确保它是一个 numpy 数组
        state_tensor = torch.tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)  # 再转换为张量
        state_tensor = state_tensor.view(1,-1)
        print(f"s shape{state_tensor.shape}")
        with torch.no_grad():
            action = model(state_tensor).cpu().numpy()  # 获取模型动作
            print('33333333333333333333333', action)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            action = action.reshape((10, 1))
            print(f"a shape{action.shape}")
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Total reward: {total_reward}")


if __name__ == "__main__":
    # 设置你的 Gym 环境名和模型路径
    MODEL_PATH = "./trained_models/ppo/Cassie-v0/e7fc15-seed42/actor.pt"  # 训练好的模型文件路径
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    env = gym.make("me5418-Cassie-v0")

    # 加载训练好的 PPO 模型
    model = load_model(MODEL_PATH, device)

    # 开始可视化
    visualize_policy(env, model, device)

    # 关闭环境
    env.close()