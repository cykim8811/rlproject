"""
The original code is from https://github.com/vwxyzjn/ppo-implementation-details
Changed parts are commented.
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="AlienNoFrameskip-v4",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=2,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=False,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(gym_id, seed, idx, capture_video, run_name, render=False):
    def thunk():
        if render:
            env = gym.make(gym_id, render_mode="human")
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.critic_variance = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def get_value_variance(self, x):
        return self.critic_variance(self.network(x / 255.0))


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

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
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    render_env = make_env(args.gym_id, 0, 0, args.capture_video, run_name, render=False)()

    # ========== CHANGE ==========
    # The Predicted UCB Methodology uses two models: target agent, behavioral agent.
    # ============================
    target_policy = Agent(envs).to(device)
    behavior_policy = Agent(envs).to(device)
    optimizer = optim.Adam(list(target_policy.parameters()) + list(behavior_policy.parameters()), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = behavior_policy.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            # ========== CHANGE ==========
            # The Predicted UCB Methodology uses two models: target agent, behavioral agent.
            # ============================
            next_value = behavior_policy.get_value(next_obs).reshape(1, -1)
            target_next_value = target_policy.get_value(next_obs).reshape(1, -1)
            
            returns = torch.zeros_like(rewards).to(device)
            target_returns = torch.zeros_like(rewards).to(device)
            # variance_reward = target policy's value variance of obs
            variance_reward = target_policy.get_value_variance(obs.reshape((-1,) + envs.single_observation_space.shape)).reshape(-1, args.num_envs)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    next_return = next_value
                    target_next_return = target_next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_return = returns[t + 1]
                    target_next_return = target_returns[t + 1]
                # ========== CHANGE ==========
                # behavioral agent is trained with r + k * sigma,
                # where sigma is the square root of predicted variance.
                # which is the calculated predicted UCB.
                # ============================
                returns[t] = rewards[t] + variance_reward[t].sqrt() # + args.gamma * nextnonterminal * next_return
                target_returns[t] = rewards[t] + args.gamma * nextnonterminal * target_next_return
            advantages = returns - values
            target_values = target_policy.get_value(obs.reshape((-1,) + envs.single_observation_space.shape)).reshape(-1, args.num_envs)
            target_advantages = target_returns - target_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_target_values = target_values.reshape(-1)

        # calculate b_target_logprobs
        with torch.no_grad():
            _, b_target_logprobs, _, _ = target_policy.get_action_and_value(b_obs, b_actions.long())
        # Importance sampling weights
        
        # ========== CHANGE ==========
        # Importance sampling calculated for target agent update.
        # ============================
        importance_sampling_weights = torch.exp(b_target_logprobs - b_logprobs).reshape(-1).detach()

        b_target_value_error = target_advantages.reshape(-1).detach() ** 2

        b_target_advantages = target_advantages.reshape(-1).detach()
        b_target_values = target_policy.get_value(b_obs).reshape(-1).detach()
        # Calculate b_target_returns from b_target_advantages
        b_target_returns = b_target_advantages + b_target_values

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        target_clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = behavior_policy.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                _, target_newlogprob, target_entropy, target_newvalue = target_policy.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                target_value_variance = target_policy.get_value_variance(b_obs[mb_inds]).reshape(-1).detach()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                target_logratio = target_newlogprob - b_target_logprobs[mb_inds]
                target_ratio = target_logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    target_old_approx_kl = (-target_logratio).mean()
                    target_approx_kl = ((target_ratio - 1) - target_logratio).mean()
                    target_clipfracs += [((target_ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                target_mb_advantages = b_target_advantages[mb_inds]
                if args.norm_adv:
                    target_mb_advantages = (target_mb_advantages - target_mb_advantages.mean()) / (target_mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                target_pg_loss1 = -target_mb_advantages * target_ratio
                target_pg_loss2 = -target_mb_advantages * torch.clamp(target_ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                target_pg_loss = torch.max(target_pg_loss1, target_pg_loss2).mean()


                target_variance_loss = 0.5 * (target_value_variance - b_target_value_error[mb_inds]).pow(2).mean()

                # Value loss
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

                target_newvalue = target_newvalue.view(-1)
                target_v_loss_unclipped = (target_newvalue - b_target_returns[mb_inds]) ** 2
                # ========== CHANGE ==========
                # Importance sampling multiplied to target value loss
                # ============================
                target_v_loss_unclipped *= importance_sampling_weights[mb_inds]
                target_v_clipped = b_values[mb_inds] + torch.clamp(
                    target_newvalue - b_values[mb_inds],
                    -args.clip_coef,
                    args.clip_coef,
                )
                target_v_loss_clipped = (target_v_clipped - b_target_returns[mb_inds]) ** 2
                target_v_loss_max = torch.max(target_v_loss_unclipped, target_v_loss_clipped)
                target_v_loss = 0.5 * target_v_loss_max.mean()

                entropy_loss = entropy.mean() + target_entropy.mean()
                loss = (pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef) + \
                        (target_pg_loss - args.ent_coef * target_entropy.mean() + target_v_loss * args.vf_coef) + \
                        target_variance_loss


                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(behavior_policy.parameters(), args.max_grad_norm)
                nn.utils.clip_grad_norm_(target_policy.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
            


        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", target_v_loss.mean().item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", target_entropy.mean().item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


        # ========== CHANGE ==========
        # Test target policy
        # ============================
        render_obs = render_env.reset()
        render_score = []
        current_render_score = 0
        while True:
            # Calculate Action with target policy
            render_action, _, _, _ = target_policy.get_action_and_value(torch.Tensor(np.array(render_obs)).to(device).unsqueeze(0))
            next_render_obs, render_reward, done, _ = render_env.step(render_action.cpu().item())
            current_render_score += render_reward
            # render_env.render()
            render_obs = next_render_obs
            if done:
                render_obs = render_env.reset()
                render_score.append(current_render_score)
                current_render_score = 0
                if len(render_score) == 20:
                    break
            
        writer.add_scalar("charts/render_score", np.mean(render_score), global_step)

        print(f"Render Score: {np.mean(render_score)}")


        # if epoch % 3 == 0:
        #     # Render
        #     render_obs = render_env.reset()
        #     while True:
        #         # Calculate Action with target policy
        #         render_action, _, _, _ = target_policy.get_action_and_value(torch.Tensor(render_obs).to(device).unsqueeze(0))
        #         next_render_obs, _, done, _ = render_env.step(render_action.cpu().item())
        #         render_env.render()
        #         render_obs = next_render_obs
        #         if done:
        #             break
        #     render_obs = render_env.reset()
        #     while True:
        #         # Calculate Action with behavior policy
        #         render_action, _, _, _ = behavior_policy.get_action_and_value(torch.Tensor(render_obs).to(device).unsqueeze(0))
        #         next_render_obs, _, done, _ = render_env.step(render_action.cpu().item())
        #         render_env.render()
        #         render_obs = next_render_obs
        #         if done:
        #             break

    envs.close()
    writer.close()
