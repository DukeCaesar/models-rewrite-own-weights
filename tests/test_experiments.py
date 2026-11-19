import json
from self_rewrite.experiments import session_schedule,write_plan,deployment_history
from self_rewrite.config import preset
from self_rewrite.session import actions


def test_schedule_all_sizes_and_independent_seeds(tmp_path):
    jobs=session_schedule()
    assert len(jobs)==17 and len({j['name'] for j in jobs})==17
    assert sum(j['repair'] for j in jobs)==4
    second=[j for j in jobs if j['protocol']==2]
    assert [j['turn_seed'] for j in second]==list(range(8))
    for job in jobs:
        config=preset(**{k:v for k,v in job.items() if k!='name'})
        assert config.passes==20 and config.turns==160
    write_plan(tmp_path/'plan.json')
    for job,submitted in zip(second,[True,False,True]):
        path=tmp_path/'runs'/job['name'];path.mkdir(parents=True)
        (path/'state.json').write_text(json.dumps({'phase':'done','result':{'submitted':submitted}}))
    history=deployment_history(tmp_path/'runs')
    assert [h['model_in_use'] for h in history]==[second[0]['name'],second[0]['name'],second[2]['name']]


def test_incomplete_reasoning_never_runs_draft_code():
    assert not actions('<think>Maybe try\n```python\nprint(1)\n```')
    assert actions('<think>draft\n```python\nprint(1)\n```</think>\n```python\nprint(2)\n```')==[('python','print(2)\n')]
