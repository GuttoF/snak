import torch
import numpy as np
from copy import deepcopy
from ppo import ActorPPO, CriticAdv, ActorDiscretePPO


class AgentPPO:
    def __init__(self):
        super().__init__()
        self.ratio_clip = 0.2  # ratio.clamp(1 - clip, 1 + clip)
        self.lambda_entropy = 0.02
        # pode ser 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.98
        self.get_reward_sum = self.get_reward_sum_raw
        self.state = None
        self.device = None
        self.criterion = torch.nn.SmoothL1Loss()
        self.act = ActorPPO(0, 0, 0)  # inicializar com valores fictícios
        self.act_optimizer = None
        self.cri = CriticAdv(0, 0)  # inicializar com valores fictícios
        self.cri_optimizer = None
        self.cri_target = CriticAdv(0, 0)  # inicializar com valores fictícios

    def init(self, net_dim, state_dim, action_dim,
            learning_rate=1e-4, if_use_gae=False):
        self.device = torch.device("cpu")

        self.get_reward_sum = self.get_reward_sum_gae \
            if if_use_gae else self.get_reward_sum_raw
        self.act = ActorPPO(net_dim, state_dim, action_dim).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri)
        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(
            self.act.parameters(),
            lr=learning_rate
        )
        self.cri_optimizer = torch.optim.Adam(
            self.cri.parameters(),
            lr=learning_rate
        )

    def select_action(self, state):
        states = torch.as_tensor(
            (state,),
            dtype=torch.float32,
            device=self.device
        )
        actions, noises = self.act.get_action(states)

        return actions[0].cpu().detach().numpy(), noises[0].cpu().detach().numpy()

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = []
        state = self.state
        for _ in range(target_step):
            action, noise = self.select_action(state)
            next_state, reward, done, _ = env.step(np.tanh(action))
            other = (
                reward * reward_scale,
                0.0 if done else gamma,
                *action,
                *noise
            )
            trajectory_list.append((state, other))
            state = env.reset() if done else next_state

        self.state = state

        return trajectory_list

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau):
        buffer.update_now_len()
        buf_len = buffer.now_len
        buf_state, buf_action, buf_r_sum, buf_logprob, buf_advantage = \
            self.prepare_buffer(buffer)
        buffer.empty_buffer()
        obj_critic = torch.tensor(0.0, device=self.device)
        obj_actor = torch.tensor(0.0, device=self.device)
        logprob = torch.tensor(0.0, device=self.device)
        for _ in range(int(buf_len / batch_size * repeat_times)):
            indices = torch.randint(
                buf_len,
                size=(batch_size,),
                requires_grad=False,
                device=self.device
            )
            state = buf_state[indices]
            action = buf_action[indices]
            r_sum = buf_r_sum[indices]
            logprob = buf_logprob[indices]
            advantage = buf_advantage[indices]
            new_logprob, obj_entropy = self.act.get_logprob_entropy(
                state,
                action
            )  # é o obj_actor
            ratio = (new_logprob - logprob.detach()).exp()
            surrogate1 = advantage * ratio
            surrogate2 = advantage * \
                ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(surrogate1, surrogate2).mean()
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy
            self.optim_update(self.act_optimizer, obj_actor)
            # a rede crítica prevê a soma de recompensas (valor Q) do estado
            value = self.cri(state).squeeze(1)
            obj_critic = self.criterion(value, r_sum) / (r_sum.std() + 1e-6)
            self.optim_update(self.cri_optimizer, obj_critic)
            self.soft_update(
                self.cri_target,
                self.cri,
                soft_update_tau
            ) if self.cri_target is not self.cri else None
        return obj_critic.item(), obj_actor.item(), logprob.mean().item()

    def prepare_buffer(self, buffer):
        buf_len = buffer.now_len
        # calcular recompensa reversa
        reward, mask, action, a_noise, state = buffer.sample_all()
        bs = 2 ** 10  # defina um 'BatchSize' menor quando estiver sem memória GPU.
        value = torch.cat(
            [self.cri_target(state[i:i + bs]).detach()
            for i in range(0, state.size(0), bs)],
            dim=0
        )
        logprob = self.act.get_old_logprob(action, a_noise)
        pre_state = torch.as_tensor(
            (self.state,),
            dtype=torch.float32,
            device=self.device
        )
        pre_r_sum = self.cri(pre_state).detach()
        r_sum, advantage = self.get_reward_sum(
            self,
            buf_len,
            reward,
            mask,
            value,
            pre_r_sum
        )

        return state, action, r_sum, logprob, advantage

    @staticmethod
    def get_reward_sum_raw(
        self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum # type: ignore
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buf_r_sum = torch.empty(
            buf_len,
            dtype=torch.float32,
            device=self.device
        )  # recompensa total
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]

        buf_advantage = buf_r_sum - (buf_mask * buf_value.squeeze(1))
        buf_advantage = (buf_advantage - buf_advantage.mean()
                        ) / (buf_advantage.std() + 1e-5)

        return buf_r_sum, buf_advantage

    @staticmethod
    def get_reward_sum_gae(
        self, buf_len, buf_reward, buf_mask, buf_value, pre_r_sum # type: ignore
    ) -> tuple[torch.Tensor, torch.Tensor]:
        buf_r_sum = torch.empty(
            buf_len,
            dtype=torch.float32,
            device=self.device
        )  # valor da recompensa anterior
        buf_advantage = torch.empty(
            buf_len,
            dtype=torch.float32,
            device=self.device
        )  # valor da vantagem
        pre_advantage = 0  # valor da vantagem anterior
        for i in range(buf_len - 1, -1, -1):
            buf_r_sum[i] = buf_reward[i] + buf_mask[i] * pre_r_sum
            pre_r_sum = buf_r_sum[i]
            buf_advantage[i] = buf_reward[i] + buf_mask[i] * \
                (pre_advantage - buf_value[i])
            pre_advantage = buf_value[i] + \
                buf_advantage[i] * self.lambda_gae_adv

        buf_advantage = (buf_advantage - buf_advantage.mean()
                        ) / (buf_advantage.std() + 1e-5)

        return buf_r_sum, buf_advantage

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data.__mul__(tau) + tar.data.__mul__(1 - tau))


class AgentDiscretePPO(AgentPPO):
    def init(self, net_dim, state_dim, action_dim,
            learning_rate=1e-4, if_use_gae=False):
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        self.get_reward_sum = self.get_reward_sum_gae \
            if if_use_gae else self.get_reward_sum_raw
        self.act = ActorDiscretePPO(
            net_dim,
            state_dim,
            action_dim
        ).to(self.device)
        self.cri = CriticAdv(net_dim, state_dim).to(self.device)
        self.cri_target = deepcopy(self.cri) if self.cri_target else self.cri
        self.criterion = torch.nn.SmoothL1Loss()
        self.act_optimizer = torch.optim.Adam(
            self.act.parameters(),
            lr=learning_rate
        )
        self.cri_optimizer = torch.optim.Adam(
            self.cri.parameters(),
            lr=learning_rate
        )

    def explore_env(self, env, target_step, reward_scale, gamma):
        trajectory_list = []
        state = self.state
        for _ in range(target_step):
            a_int, a_prob = self.select_action(state)
            next_state, reward, done, _ = env.step(int(a_int))
            other = (
                reward * reward_scale,
                0.0 if done else gamma,
                a_int,
                *a_prob
            )
            trajectory_list.append((state, other))
            state = env.reset() if done else next_state

        self.state = state

        return trajectory_list
