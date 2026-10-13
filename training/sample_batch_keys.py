"""할당 헤드의 학습 신호를 담는 SampleBatch / Postprocessing 커스텀 키.

RLlib SampleBatch 는 임의 문자열 키를 허용하므로 문자열 상수로 둔다.
primitive 헤드는 RLlib 기본 키(REWARDS, VF_PREDS, ACTION_LOGP, ADVANTAGES, VALUE_TARGETS)를 쓴다.
"""

REWARDS_ALLOC = "rewards_alloc"            # env info -> per-agent, postprocess 에서 채움
VF_PREDS_ALLOC = "vf_preds_alloc"          # rollout 시 extra_action_out
ACTION_LOGP_ALLOC = "action_logp_alloc"    # rollout 시 extra_action_out
ADVANTAGES_ALLOC = "advantages_alloc"      # postprocess
VALUE_TARGETS_ALLOC = "value_targets_alloc"

# 정책이 실제로 할당을 고른 (lock 강제 아님) 에이전트 = 1. alloc policy loss 마스크 (env info "alloc_valid")
ALLOC_VALID = "alloc_valid"
