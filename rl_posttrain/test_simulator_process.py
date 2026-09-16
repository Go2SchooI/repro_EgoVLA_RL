import json,os,subprocess,sys
from pathlib import Path
import pytest
from rl_posttrain.simulator_process import run_simulator


def environment(tmp_path):
    return dict(os.environ,RUN_DIR=str(tmp_path/"run"),RESULT_PATH=str(tmp_path/"result.txt"),
                RL_COLLECT_REPLAY_PATH=str(tmp_path/"replay.npz"),COUNTER=str(tmp_path/"counter"))


@pytest.mark.parametrize("nested_result",[False,True])
def test_device_lost_retries_preserves_partial_outputs_and_same_seed(tmp_path,nested_result):
    env=environment(tmp_path);env["RL_SAC_SEED"]="111"
    if nested_result:env["RESULT_PATH"]=str(tmp_path/"run"/"result.txt")
    script="""import os,time
from pathlib import Path
p=Path(os.environ['COUNTER']);count=int(p.read_text())+1 if p.exists() else 1;p.write_text(str(count))
assert os.environ['RL_SAC_SEED']=='111'
Path(os.environ['RESULT_PATH']).write_text('partial' if count==1 else 'complete')
Path(os.environ['RL_COLLECT_REPLAY_PATH']).write_text('partial' if count==1 else 'complete')
if count==1:
 print('VkResult: ERROR_DEVICE_LOST',flush=True);time.sleep(30)
"""
    result=run_simulator([sys.executable,"-c",script],env=env,poll_interval=.02)
    assert result.returncode==0
    assert Path(env['RESULT_PATH']).read_text()=='complete'
    assert Path(env['RL_COLLECT_REPLAY_PATH']).read_text()=='complete'
    archived=list((tmp_path/'simulator_retries').glob('*/replay.npz'))
    assert len(archived)==1 and archived[0].read_text()=='partial'
    assert len(json.loads((tmp_path/'simulator_attempts.json').read_text()))==2


def test_non_device_error_is_not_retried(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_simulator([sys.executable,'-c','raise SystemExit(3)'],env=environment(tmp_path),poll_interval=.02)
    assert len(json.loads((tmp_path/'simulator_attempts.json').read_text()))==1


def test_idle_timeout_bounds_attempts(tmp_path):
    with pytest.raises(subprocess.CalledProcessError):
        run_simulator([sys.executable,'-c','import time;time.sleep(30)'],env=environment(tmp_path),
                      poll_interval=.02,idle_timeout=.1,max_attempts=2)
    assert len(list((tmp_path/'simulator_retries').glob('*/failure.json')))==2
