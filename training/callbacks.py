"""에피소드 지표를 RLlib custom_metrics 로 기록하는 콜백 (TensorBoard / progress.csv / reward_curve.png).

RLlib 의 episode_reward 는 env 가 넘기는 스칼라(할당 + primitive, UAV 전체 합)라서 두 네트워크를 구분할 수 없다.
여기서 info 의 계층별 보상과 임무 지표를 따로 모아 result["custom_metrics"] 에 넣는다
(progress.csv 에는 custom_metrics/rescue_rate_mean 처럼 _mean / _min / _max 로 기록된다).

    reward_alloc     : 에피소드 동안 할당 보상 합, UAV 평균  -> 할당 헤드가 최대화하는 값
    reward_primitive : 에피소드 동안 primitive 보상 합, UAV 평균 -> primitive 헤드가 최대화하는 값
    rescue_rate      : 구조 수 / N_T
    expire_rate      : 만료 수 / N_T
    discover_rate    : 발견 수 / N_T
    collisions       : 에피소드 누적 UAV 충돌 횟수 (충돌한 UAV 수 합)
    explored         : 에피소드 끝 팀 탐색 비율 (도달 가능한 빈 칸 중 팀이 아는 칸)
    episode_len      : 에피소드 길이 (step)

env 는 4대를 묶은 단일 gym.Env 라 RLlib 1.3 이 dummy agent 하나로 감싸고, episode.last_info_for() 로 그 step 의 info 를 읽는다.
training/train.py 가 config["callbacks"] 로 지정한다.
"""
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks


class RescueCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        episode.user_data["reward_alloc"] = 0.0
        episode.user_data["reward_primitive"] = 0.0

    def on_episode_step(self, *, worker, base_env, episode, env_index=None, **kwargs):
        info = episode.last_info_for() or {}
        for key in ("reward_alloc", "reward_primitive"):
            if key in info:
                episode.user_data[key] += float(np.mean(info[key]))

    def on_episode_end(self, *, worker, base_env, policies, episode, env_index=None, **kwargs):
        m = episode.custom_metrics
        for key in ("reward_alloc", "reward_primitive"):
            m[key] = episode.user_data[key]
        m["episode_len"] = episode.length
        info = episode.last_info_for() or {}
        if "rescued_total" in info:
            n_targets = max(1, info["rescued_total"] + info["expired_total"] + info["active"])   # = N_T
            m["rescue_rate"] = info["rescued_total"] / n_targets
            m["expire_rate"] = info["expired_total"] / n_targets
            m["discover_rate"] = info["discovered_total"] / n_targets
            m["collisions"] = info["collisions_total"]
            m["explored"] = info["explored"]
