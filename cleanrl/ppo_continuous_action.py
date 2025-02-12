import os
import random
import time
import sys
import math

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
import reward_predictor

from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from ppo_config import Args
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from policy_network import PPOPolicy

sys.path.append('/Users/maxi/Developer/Python/rlhf_f')
import torch.nn as nn
import torch.optim as optim



args = Args()


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="human")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)

        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = self.update_mean_var_count(
            self.mean, self.var, self.count,
            batch_mean, batch_var, batch_count
        )

    @staticmethod
    def update_mean_var_count(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.total_sample_size = int(args.sample_size * args.size_segment)
    args.total_sample_update = int(args.sample_update * args.size_segment)
    args.batch_size = int(args.num_envs * args.total_sample_update)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_sample_size // args.total_sample_update
    updates_after_stp = (args.total_timesteps // args.num_iterations) // args.num_envs
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    policy_net = PPOPolicy(envs).to(device)
    reward_net = reward_predictor.RewardPredictor(envs).to(device)

    reward_predictor_optimizer = optim.Adam(
        reward_net.parameters(), lr=args.learning_rate, eps=1e-3
    )
    policy_optimizer = optim.Adam(
        policy_net.actor_mean.parameters(), lr=args.learning_rate, eps=1e-3
    )

    obs = torch.zeros(
        (updates_after_stp, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (updates_after_stp, args.num_envs) + envs.single_action_space.shape
    ).to(device)

    logprobs = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    rewards = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    dones = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    values = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    advantages = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    preferences = []

    def create_segment(segment_list):
        segment_data = []
        for _ in range(70):
            if len(segment_list) < args.size_segment:
                break
            start_index = random.randint(0, len(segment_list) - args.size_segment)
            segment_data.append(segment_list[start_index:start_index + args.size_segment])
        return segment_data

    def generate_preference_label(segment1, segment2):
        segment1_reward_sum = torch.sum(torch.stack([s[2] for s in segment1]))
        segment2_reward_sum = torch.sum(torch.stack([s[2] for s in segment2]))

        if segment1_reward_sum > segment2_reward_sum:
            return 1
        elif segment1_reward_sum < segment2_reward_sum:
            return 0
        else:
            return None

    def train_reward_predictor(preferences, writer, global_step):
        reward_net.train()  # ensure we are in training mode
        reward_net.zero_grad()
          # use a list to collect losses

        for i in range(1, 7):
            loss_list = []
            chunk = preferences[-(i + args.sample_update):-i]
            if global_step % 100  == 0:
                num_items_to_sample = random.randint(6, min(15, len(preferences)))
                random_items = random.sample(preferences, num_items_to_sample)
                chunk.extend(random_items)
                print(f"random_items={random_items}")
            for segment1, segment2, label in chunk:
                segment1_tensor_list = []
                for s in segment1:
                    segment1_tensor_list.append(torch.cat([s[0], s[1]], dim=-1).to(device))
                segment1_tensors = torch.stack(segment1_tensor_list)

                segment2_tensor_list = []
                for s in segment2:
                    segment2_tensor_list.append(torch.cat([s[0], s[1]], dim=-1).to(device))
                segment2_tensors = torch.stack(segment2_tensor_list)

                # Ensure gradients are enabled for the loss computation.
                with torch.set_grad_enabled(True):
                    segment1_reward = reward_net.predict_reward(segment1_tensors)
                    segment2_reward = reward_net.predict_reward(segment2_tensors)

                    segment1_reward_sum = torch.sum(segment1_reward)
                    segment2_reward_sum = torch.sum(segment2_reward)
                    epsilon = 1e-8

                    diff1 = segment1_reward_sum - segment2_reward_sum
                    diff2 = segment2_reward_sum - segment1_reward_sum
                    better1 = torch.sigmoid(diff1)
                    better2 = torch.sigmoid(diff2)

                    if torch.isnan(better1).any():
                        print("NaN detected in probabilities")
                        writer.add_scalar("nan_detected", 1, global_step)
                        continue

                    if torch.isnan(better2).any():
                        print("NaN detected in probabilities")
                        writer.add_scalar("nan_detected", 1, global_step)
                        continue

                    # Clamp the probabilities and reassign
                    better1 = torch.clamp(better1, min=epsilon, max=1 - epsilon)
                    better2 = torch.clamp(better2, min=epsilon, max=1 - epsilon)

                    # Compute loss; ensure label is a tensor if needed.
                    loss = torch.log(better1) * label + (1 - label) * torch.log(better2)
                    loss_list.append(loss)

        if loss_list:
            total_loss = -torch.stack(loss_list).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_net.parameters(), max_norm=1.0)
        
        return preferences

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    pretraining_steps = int(args.total_timesteps * 0.25)

    while global_step <= pretraining_steps:
        for step in range(0, updates_after_stp):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _ = policy_net.get_action_and_value(next_obs)
                action_nn_obs = torch.cat((action, next_obs), dim=-1)
                value = reward_net.forward(action_nn_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)

            next_obs_tensor = torch.from_numpy(next_obs).to(device).float()
            action_n_obs = torch.cat((action, next_obs_tensor), dim=-1)
            rewards[step] = reward_net.predict_reward(action_n_obs).view(-1)

            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            segment = (
                torch.tensor(obs[step], dtype=torch.float32),
                torch.tensor(action, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
            )
            reward_predictor.segment_list.append(segment)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        """ print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        ) """
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        segment_data = create_segment(reward_predictor.segment_list)
        #print(f"segment_data={segment_data}")

        for i in range(len(segment_data)//2):
            if len(segment_data) < 2:
                break

            segment1 = segment_data[i if i % 2 == 0 else i+1]
            segment2 = segment_data[i+1 if i % 2 == 0 else i]
            preference_label = generate_preference_label(segment1, segment2)

            if preference_label is not None:
                preferences.append((segment1, segment2, preference_label))
        print(f"preferences={preferences}")
        preferences = train_reward_predictor(preferences, writer, global_step)

    logprobs = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    rewards = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    dones = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    values = torch.zeros((updates_after_stp, args.num_envs)).to(device)
    advantages = torch.zeros((updates_after_stp, args.num_envs)).to(device)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations*2 + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations*2
            lrnow = frac * args.learning_rate
            reward_predictor_optimizer.param_groups[0]["lr"] = lrnow
            for param_group in policy_optimizer.param_groups:
                param_group["lr"] = lrnow

        for step in range(0, updates_after_stp):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _ = policy_net.get_action_and_value(next_obs)
                action_nn_obs = torch.cat((action, next_obs), dim=-1)
                value = reward_net.forward(action_nn_obs)
                values[step] = value.flatten()

            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)

            next_obs_tensor = torch.from_numpy(next_obs).to(device).float()
            action_n_obs = torch.cat([action, next_obs_tensor], dim=-1)
            rewards[step] = reward_net.predict_reward(action_n_obs).view(-1)

            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

            segment = (
                torch.tensor(obs[step], dtype=torch.float32),
                torch.tensor(action, dtype=torch.float32),
                torch.tensor(reward, dtype=torch.float32),
            )
            reward_predictor.segment_list.append(segment)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            action, _, _ = policy_net.get_action_and_value(next_obs)
            action_nn_obs = torch.cat([action, next_obs], dim=-1)
            next_value = reward_net.forward(action_nn_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0

            for t in reversed(range(updates_after_stp)):
                if t == updates_after_stp - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]

                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        segment_data = create_segment(reward_predictor.segment_list)

        for i in range(len(segment_data)//2):
            if len(segment_data) < 2:
                break

            segment1 = segment_data[i if i % 2 == 0 else i+1]
            segment2 = segment_data[i+1 if i % 2 == 0 else i]
            preference_label = generate_preference_label(segment1, segment2)

            if preference_label is not None:
                preferences.append((segment1, segment2, preference_label))

        preferences = train_reward_predictor(preferences, writer, global_step)
        writer.add_scalar("length_of_preferences", len(preferences), global_step)

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        if global_step > pretraining_steps:
            b_inds = np.arange(args.batch_size)
            clipfracs = []

            for epoch in range(args.sample_update):
                np.random.shuffle(b_inds)

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy = policy_net.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    newvalue = reward_net.forward(torch.cat([b_actions[mb_inds], b_obs[mb_inds]], dim=-1))
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (
                            mb_advantages - mb_advantages.mean()
                        ) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)

                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    policy_optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(policy_net.parameters(), args.max_grad_norm)
                    policy_optimizer.step()

                    with torch.no_grad():
                        policy_net.actor_logstd.data.clamp_(-5.0, 2.0)

        reward_predictor.segment_list = []
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )

        for idx, episodic_return in enumerate(episodic_returns):
            print(f"Episode {idx}: Return = {episodic_return.item():.2f}")
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = (
                f"{args.hf_entity}/{repo_name}"
                if args.hf_entity
                else repo_name
            )
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval"
            )

    envs.close()
    writer.close()
