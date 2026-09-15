import json
from pathlib import Path

import pytest

from rl_posttrain.queue_sac import predecessor_complete
from rl_posttrain.run_sac import atomic, sha


def test_queue_requires_completed_run_and_all_dual_evaluation_receipts(tmp_path):
    assert not predecessor_complete(tmp_path)
    for stage in ["evaluating","failed","collecting"]:
        atomic(tmp_path/"status.json",dict(stage=stage,online_episode=150,target_episode=150))
        assert not predecessor_complete(tmp_path)
    atomic(tmp_path/"status.json",dict(stage="completed",online_episode=150,target_episode=150))
    assert not predecessor_complete(tmp_path)
    actor=tmp_path/"actor.pt";actor.write_bytes(b"checkpoint")
    result=tmp_path/"results.txt";result.write_text("verified results")
    mode=dict(groups={"all":{"n":96}},scenes=[dict(result_path=str(result),result_sha256=sha(result))]*6)
    for episode in [50,100,150]:
        atomic(tmp_path/f"eval/episode_{episode:04d}/verification.json",dict(episode=episode,
            actor_checkpoint=str(actor),actor_sha256=sha(actor),modes=dict(deterministic=mode,stochastic=mode)))
    assert predecessor_complete(tmp_path)
    result.write_text("changed")
    with pytest.raises(ValueError,match="result changed"):
        predecessor_complete(tmp_path)
