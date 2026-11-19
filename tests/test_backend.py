import pytest
import torch
from safetensors.torch import save_file
from self_rewrite.backend import TransformersBackend, Problem
from self_rewrite.checkpoint import Checkpoint
from self_rewrite.edits import EditStore
from self_rewrite.config import SessionConfig


@pytest.fixture
def backend(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    torch.manual_seed(1)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                  eos_token_id=2, bos_token_id=1, pad_token_id=0))
    checkpoint_path = tmp_path/'base'; model.save_pretrained(checkpoint_path)
    tokens = Tokenizer(WordLevel({'[PAD]':0,'[BOS]':1,'[EOS]':2,'[UNK]':3,'one':4,'two':5,'three':6},unk_token='[UNK]'))
    tokens.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokens, unk_token='[UNK]',pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]')
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    key = 'model.layers.0.mlp.down_proj.weight'
    damage = tmp_path/'damage.safetensors'
    save_file({key: torch.zeros_like(model.state_dict()[key])},damage)
    checkpoint = Checkpoint(checkpoint_path,damage)
    edits = EditStore(tmp_path/'edited',checkpoint.files,checkpoint.load)
    config = SessionConfig(max_tokens=16,editor_tokens=16)
    return TransformersBackend(model,tokenizer,checkpoint,edits,config)


def test_weights_restore_even_on_failure(backend):
    name = 'model.layers.0.mlp.down_proj.weight'
    clean = backend.model.state_dict()[name].clone()
    with backend.weights('original'):
        assert not backend.model.state_dict()[name].count_nonzero()
    assert torch.equal(backend.model.state_dict()[name],clean)
    edit = torch.ones_like(clean); edit[0] = 2
    backend.edits.save_edit(name,edit)
    with pytest.raises(RuntimeError):
        with backend.weights('edited'):
            assert torch.equal(backend.model.state_dict()[name],edit)
            raise RuntimeError('inference failed')
    assert torch.equal(backend.model.state_dict()[name],clean)
    backend.edits.clear()
    with backend.weights('edited'):
        assert not backend.model.state_dict()[name].count_nonzero()
    assert torch.equal(backend.model.state_dict()[name],clean)


def test_real_forward_hidden_and_generation(backend):
    text = 'one two three one'
    with backend.weights('original'), torch.no_grad():
        result = backend.model(**backend.tokenizer(text,return_tensors='pt'),output_hidden_states=True)
    actual = backend.read([text],layers=[0],positions='all')[0]['0']
    assert torch.allclose(actual,result.hidden_states[1][0])
    mean = backend.read([text],layers=[0],answer_starts=[2])[0]['0']
    assert torch.allclose(mean,actual[2:].mean(0))
    expected = result.logits[0,:-1].log_softmax(-1).gather(-1,torch.tensor([5,6,4])[:,None]).squeeze(-1)
    probs = backend.read([text],layers=[],kind='logprobs',positions='all')[0]['logprobs']
    assert torch.allclose(probs,expected,atol=1e-6)
    rows = backend.generate([Problem('1','one two','3')],samples=2)
    again = backend.generate([Problem('1','one two','3')],samples=2)
    assert rows == again and len(rows) == 2
    assert rows[0].seed != rows[1].seed
    assert all(0 < r.tokens <= 16 for r in rows)
    assert backend.reply([{'role':'user','content':'one two'}],0)


def test_checkpoint_export_preserves_damage_and_edit(backend,tmp_path):
    key = 'model.layers.1.mlp.down_proj.weight'
    value = backend.checkpoint.load(key).float(); value[0] = 1
    backend.edits.save_edit(key,value)
    output = tmp_path/'export'
    backend.checkpoint.export(output,backend.edits)
    exported = Checkpoint(output)
    assert torch.equal(exported.load(key),value.bfloat16())
    assert exported.load('model.layers.0.mlp.down_proj.weight').count_nonzero() == 0
    assert set(exported.files) == set(backend.checkpoint.files)
    assert len(backend.checkpoint.output_matrices(0)) == 2
    assert backend.checkpoint.shapes()[key] == list(value.shape)
    with pytest.raises(ValueError): backend.checkpoint.export(output,backend.edits)


def test_mlp_reads_intermediate_neurons(backend):
    result = backend.read(['one two three'], layers=[0], kind='mlp',positions='all')
    assert result[0]['0'].shape == (3,32)
    with pytest.raises(ValueError): backend.read(['one'],layers=[99])
    # A failing read must not leave edited or damaged weights loaded.
    assert not backend._active


def test_forward_requests_budget_and_artifacts(backend,tmp_path):
    import json
    from self_rewrite.forward import ForwardRunner
    runner = ForwardRunner(backend,[Problem('p','one two','1')],tmp_path/'work')
    for bad in [dict(kind='generate',ids=['heldout']),
                dict(kind='generate',ids=['p'],samples=257,weights='both'),
                dict(kind='generate',ids=['p'],max_tokens=15),
                dict(kind='read',texts=['one'],weights='editor')]:
        with pytest.raises(ValueError): runner.run(bad)
    assert runner.used == 0
    observation = runner.run(dict(kind='generate',ids=['p'],weights='both'))
    assert runner.used == 1 and observation['passes_used'] == 1
    rows = [json.loads(line) for line in (runner.work/observation['artifact']).read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['seed'] == rows[1]['seed']
    observation = runner.run(dict(kind='read',texts=['one two'],layers=[0]))
    assert runner.used == 2
    saved = torch.load(runner.work/observation['artifact'],weights_only=True)
    assert saved[0]['0'].shape == (16,)
    runner.used = backend.config.passes
    with pytest.raises(ValueError): runner.run(dict(kind='generate',ids=['p']))


def test_persistent_worker_capabilities_and_timeout(backend,tmp_path):
    from self_rewrite.worker import CodeWorker
    worker=CodeWorker(backend.checkpoint,backend.edits,[],tmp_path/'work',timeout=10)
    try:
        result=worker.run("x=7\nprint(x)")
        assert result['ok'] and result['output']=='7\n',result
        assert worker.run('print(x+1)')['output']=='8\n'
        name='model.layers.0.mlp.down_proj.weight'
        assert worker.run(f"w=load_original('{name}')\nprint(w.count_nonzero().item())")['output']=='0\n'
        result=worker.run(f"w[0]=1\nsave_edit('{name}',w)\nwrite_notes('saved')")
        assert result['ok'],result
        assert name in backend.edits.list_edits()
        assert (tmp_path/'work'/'notes.md').read_text()=='saved'
        for code in ['import os','torch.cuda.is_available()','w.backward()',"open('/etc/passwd')","load_artifact('../secret')",'print(torch.__dict__)']:
            assert not worker.run(code)['ok']
        worker.timeout=.2
        assert 'timed out' in worker.run('while True: pass')['error']
        worker.timeout=10
        assert worker.run('print(read_notes())')['output']=='saved\n'
        assert not worker.run('print(x)')['ok']
    finally: worker.close()


def test_session_runs_edits_and_resumes_completed_state(backend,tmp_path):
    from dataclasses import replace
    from self_rewrite.worker import CodeWorker
    from self_rewrite.session import Session
    backend.config=replace(backend.config,diagnosis_count=1,passes=1,turns=5)
    work=tmp_path/'session'
    problems=[Problem('p','one two','1')]
    worker=CodeWorker(backend.checkpoint,backend.edits,problems,work,timeout=10)
    name='model.layers.0.mlp.down_proj.weight'
    replies=iter([f"```python\nw=load_original('{name}')\nw[0]=1\nsave_edit('{name}',w)\n```",
                  '```forward\n{"kind":"generate","ids":["p"],"weights":"both"}\n```',
                  'Finished: keeping the saved edit.'])
    # Scripted editor tests orchestration only; forward calls execute the real model.
    backend.reply=lambda messages,turn:next(replies)
    try:
        session=Session(backend,worker,problems,work)
        result=session.run()
        assert result['submitted'] and result['passes']==1 and result['code_runs']==1
        assert (work/'census.jsonl').exists() and (work/'gen_1.jsonl').exists()
        restored=Session(backend,worker,problems,work,resume=True)
        assert restored.run()==result
        with pytest.raises(ValueError): Session(backend,worker,problems,work)
    finally: worker.close()


def test_reserved_decision_resume_does_not_repeat_check(backend,tmp_path):
    from dataclasses import replace
    from self_rewrite.session import Session
    backend.config=replace(backend.config,protocol=2,diagnosis_count=1,reserved_count=1,check_count=1)
    name='model.layers.0.mlp.down_proj.weight'
    tensor=backend.checkpoint.load(name).clone(); tensor[0]=1
    backend.edits.save_edit(name,tensor)
    session=Session(backend,None,[Problem('p','one','1')],tmp_path/'decision',reserved=[Problem('r','two','2')])
    session.phase='decision'
    session.messages=[{'role':'user','content':'Check already finished. Reply SUBMIT or WITHDRAW.'}]
    session.save()
    backend.reply=lambda messages,turn:'WITHDRAW'
    def forbidden(*args,**kwargs): raise AssertionError('Check must not run twice')
    backend.generate=forbidden
    resumed=Session(backend,None,[Problem('p','one','1')],session.work,reserved=[Problem('r','two','2')],resume=True)
    assert not resumed.run()['submitted']


def test_cli_damage_and_export(backend,tmp_path):
    from self_rewrite.cli import main,read_problems
    overlay=tmp_path/'cli-damage.safetensors'
    key='model.layers.0.mlp.down_proj.weight'
    main(['damage','--checkpoint',str(backend.checkpoint.root),'--tensor',key,
          '--row-start','0','--seed','7','--output',str(overlay)])
    damaged=Checkpoint(backend.checkpoint.root,overlay)
    assert not torch.equal(damaged.load(key),Checkpoint(backend.checkpoint.root).load(key))
    output=tmp_path/'cli-export'
    main(['export','--checkpoint',str(backend.checkpoint.root),'--damage',str(overlay),
          '--edits',str(tmp_path/'no-edits'),'--output',str(output)])
    assert torch.equal(Checkpoint(output).load(key),damaged.load(key))
    problems=tmp_path/'problems.jsonl'
    problems.write_text('{"id":"a","question":"What is 1+1?","answer":"2"}\n')
    assert read_problems(problems)[0].id=='a'


def test_heldout_plan_resumes_without_new_sampling(backend,tmp_path):
    from self_rewrite.heldout import HeldoutEvaluation
    problems=[Problem('h','one two three','1'),Problem('h2','three two','2')]
    output=tmp_path/'evaluation'
    evaluation=HeldoutEvaluation(backend,problems,[],output,samples=2,max_tokens=16,flips=100)
    assert (output/'plan.json').exists() and not (output/'report.json').exists()
    report=evaluation.run()
    assert report['complete'] and report['problems']==2
    def fail(*args,**kwargs): raise AssertionError('Completed pairs must not regenerate')
    backend.generate=fail
    resumed=HeldoutEvaluation(backend,problems,[],output,samples=2,max_tokens=16,flips=100)
    assert resumed.run()==report
    with pytest.raises(ValueError): HeldoutEvaluation(backend,problems,[],output,samples=3,max_tokens=16,flips=100)
    with pytest.raises(ValueError): HeldoutEvaluation(backend,problems,problems,output,samples=2,max_tokens=16,flips=100)
    # Changing the actual edit invalidates the precommitment even with the same path.
    name='model.layers.0.mlp.down_proj.weight'; w=backend.checkpoint.load(name); w[0]=1
    backend.edits.save_edit(name,w)
    with pytest.raises(ValueError): HeldoutEvaluation(backend,problems,[],output,samples=2,max_tokens=16,flips=100)


def test_candidate_comparison_restores_current_edit(backend,tmp_path):
    from dataclasses import replace
    from self_rewrite.forward import ForwardRunner
    backend.config=replace(backend.config,protocol=2,max_generation_answers=128)
    name='model.layers.0.mlp.down_proj.weight'
    w=backend.checkpoint.load(name);w[0]=1
    backend.edits.save_edit(name,w);backend.edits.save_candidate('first')
    w[1]=2;backend.edits.save_edit(name,w);backend.edits.save_candidate('second')
    current=backend.edits.load(name).clone()
    runner=ForwardRunner(backend,[Problem('p','one two','1')],tmp_path/'multi')
    observation=runner.run(dict(kind='generate',ids=['p'],weights='both',candidates=['first','second']))
    assert set(observation['candidates'])=={'first','second'} and runner.used==1
    assert torch.equal(backend.edits.load(name),current)
    with pytest.raises(ValueError): backend.edits.save_candidate('first')
    with pytest.raises(ValueError): backend.edits.candidate('../first')
    with pytest.raises(ValueError): runner.run(dict(kind='generate',ids=['p'],weights='both',candidates=['first','second'],samples=43))
    backend.edits.restore_candidate('first')
    assert backend.edits.load(name)[1].count_nonzero()==0


def test_protocol2_reserved_check_and_submission(backend,tmp_path):
    from dataclasses import replace
    from self_rewrite.session import Session
    from self_rewrite.worker import CodeWorker
    backend.config=replace(backend.config,protocol=2,diagnosis_count=1,reserved_count=1,check_count=1,passes=1,turns=5)
    practice=[Problem('p','one two','1')];reserved=[Problem('r','three two','2')]
    work=tmp_path/'protocol2'
    worker=CodeWorker(backend.checkpoint,backend.edits,practice,work,timeout=10)
    name='model.layers.0.mlp.down_proj.weight'
    replies=iter([f"```python\nw=load_original('{name}')\nw[0]=1\nsave_edit('{name}',w)\nsave_candidate('proposal')\n```",
                  '```forward\n{"kind":"generate","ids":["p"],"weights":"both","candidates":["proposal"]}\n```',
                  'Keep the edit.', 'SUBMIT'])
    backend.reply=lambda messages,turn:next(replies)
    try:
        result=Session(backend,worker,practice,work,reserved=reserved).run()
        assert result['submitted'] and result['passes']==1
        assert (work/'reserved_check.json').exists()
        assert not worker.run("print(load_artifact('reserved_check.json'))")['ok']
        assert worker.run("print([p['id'] for p in practice])")['output']=="['p']\n"
    finally: worker.close()


def test_resume_rejects_changed_practice_and_reconciles_committed_pass(backend,tmp_path):
    import json
    from self_rewrite.session import Session
    work=tmp_path/'recover';practice=[Problem('p','one','1')]
    session=Session(backend,None,practice,work)
    session.phase='editing';session.messages=[{'role':'user','content':'task'}]
    session.pending_action={'kind':'forward','source':'{}'}
    session.save()
    (work/'forward.jsonl').write_text(json.dumps({'observation':{'passes_used':1}})+'\n')
    resumed=Session(backend,None,practice,work,resume=True)
    assert resumed.runner.used==1
    assert any('Interrupted action' in m['content'] for m in resumed.messages)
    with pytest.raises(ValueError): Session(backend,None,[Problem('p','different','1')],work,resume=True)


def test_worker_tokenizer_capability(backend,tmp_path):
    from self_rewrite.worker import CodeWorker
    worker=CodeWorker(backend.checkpoint,backend.edits,[],tmp_path/'tokens',tokenizer=backend.tokenizer)
    try:
        result=worker.run("ids=tokenizer.encode('one two')\nprint(ids)\nprint(tokenizer.decode(ids))")
        assert result['ok'] and result['output']=='[4, 5]\none two\n'
    finally: worker.close()


def test_vllm_adapter_pairing_and_restoration_contract(backend,tmp_path):
    # Contract double: explicitly NOT evidence of a working installed vLLM runtime.
    from types import SimpleNamespace
    from safetensors.torch import load_file
    from self_rewrite.vllm_backend import VLLMBackend
    name='model.layers.0.mlp.down_proj.weight'
    clean=backend.model.state_dict()[name].clone()
    class Engine:
        def __init__(self): self.weights={name:clean};self.calls=[];self.resets=0;self.fail_once=False
        def collective_rpc(self,method,args):
            overlay=load_file(args[0]);self.weights.update(overlay)
            self.calls.append({k:v.clone() for k,v in overlay.items()})
            if self.fail_once:
                self.fail_once=False;raise RuntimeError('partial load failure')
            return [list(overlay)]
        def reset_prefix_cache(self): self.resets+=1;return True
        def generate(self,prompts,params,use_tqdm):
            if not isinstance(params,list): params=[params]*len(prompts)
            self.seeds=[p.seed for p in params]
            return [SimpleNamespace(outputs=[SimpleNamespace(text='reason</think>\\boxed{1}',token_ids=[1]*10,finish_reason='stop')]) for _ in prompts]
    engine=Engine()
    served=VLLMBackend(engine,backend.tokenizer,backend.checkpoint,backend.edits,backend.config,
                       sampling_factory=lambda **kw:SimpleNamespace(**kw))
    problems=[Problem('p','one','1')]
    a=served.generate(problems,samples=2)
    assert not engine.calls[0][name].count_nonzero()  # Damaged baseline was loaded.
    assert torch.equal(engine.weights[name],clean)
    w=backend.checkpoint.load(name);w[0]=1;backend.edits.save_edit(name,w)
    b=served.generate(problems,weights='edited',samples=2)
    assert [r.seed for r in a]==[r.seed for r in b] and all(r.correct for r in b)
    assert torch.equal(engine.calls[-2][name],w.bfloat16())
    assert torch.equal(engine.weights[name],clean) and engine.resets==4
    engine.fail_once=True
    with pytest.raises(RuntimeError): served.generate(problems,weights='edited')
    assert torch.equal(engine.weights[name],clean) and not served._active
    before=len(engine.calls)
    assert served.reply([{'role':'user','content':'one'}],0)
    assert len(engine.calls)==before  # Editor remains on clean weights.


def test_shared_baseline_is_reused_across_edits(backend,tmp_path):
    from self_rewrite.heldout import HeldoutEvaluation
    calls=[];generate=backend.generate
    def record(*args,**kwargs):
        calls.append(kwargs['weights']);return generate(*args,**kwargs)
    backend.generate=record
    problem=[Problem('h','one two','1')]
    cache=tmp_path/'baseline-cache'
    HeldoutEvaluation(backend,problem,[],tmp_path/'eval1',samples=1,max_tokens=16,flips=100,baseline_cache=cache).run()
    name='model.layers.0.mlp.down_proj.weight';w=backend.checkpoint.load(name);w[0]=1
    backend.edits.save_edit(name,w)
    HeldoutEvaluation(backend,problem,[],tmp_path/'eval2',samples=1,max_tokens=16,flips=100,baseline_cache=cache).run()
    assert calls==['original','edited','edited']


def test_staged_gates_and_two_look_precommitment(backend,tmp_path):
    from self_rewrite.staged import StagedEvaluation
    from self_rewrite.evaluation import Answer,problem_seed
    initial=[Problem(f'i{k}',f'initial {k}','1') for k in range(2)]
    confirmation=[Problem(f'c{k}',f'confirmation {k}','1') for k in range(3)]
    calls=[]
    def generate(problems,*,weights,samples=1,seed_base=0,**kwargs):
        calls.append((problems[0].id,weights,samples,seed_base))
        return [Answer(p.id,j,(problem_seed(p.id,seed_base)+j)%2**31,'',1,False,weights=='edited',False)
                for p in problems for j in range(samples)]
    backend.generate=generate
    run=StagedEvaluation(backend,initial,confirmation,[],tmp_path/'staged',two_looks=True,max_tokens=16,flips=1000)
    assert not calls and (run.root/'staged_plan.json').exists()
    report=run.run()
    assert report['gate_passed'] and len(report['cases']['success'])==2
    assert report['second_look'] is not None  # Three problem clusters cannot establish significance.
    assert report['second_look']['correct']['delta']==1.
    assert sum(samples for pid,mode,samples,seed in calls if pid=='c0' and mode=='edited')==16
    before=len(calls)
    assert run.run()==report and len(calls)==before
    with pytest.raises(ValueError): StagedEvaluation(backend,initial,confirmation,[],run.root,two_looks=False,max_tokens=16,flips=1000)
    def lose(problems,*,weights,samples=1,seed_base=0,**kwargs):
        return [Answer(p.id,j,(problem_seed(p.id,seed_base)+j)%2**31,'',1,False,weights=='original',False)
                for p in problems for j in range(samples)]
    backend.generate=lose
    losing=StagedEvaluation(backend,initial,confirmation,[],tmp_path/'losing',max_tokens=16,flips=100)
    rejected=losing.run()
    assert not rejected['gate_passed'] and rejected['confirmation'] is None
    assert not (losing.root/'confirm1').exists()


def test_explicit_overlap_audit_preserves_original_evaluation(backend,tmp_path):
    from self_rewrite.heldout import HeldoutEvaluation
    practice=[Problem('train','one two','1')]
    repeated=[Problem('heldout','one two','1')]
    with pytest.raises(ValueError): HeldoutEvaluation(backend,repeated,practice,tmp_path/'strict',samples=1,max_tokens=16)
    run=HeldoutEvaluation(backend,repeated,practice,tmp_path/'audit',samples=1,max_tokens=16,allow_practice_overlap=True)
    assert run.plan['practice_overlap_ids']==['heldout']
    with pytest.raises(ValueError): HeldoutEvaluation(backend,practice,practice,tmp_path/'sameid',samples=1,max_tokens=16,allow_practice_overlap=True)


def test_exclusion_records_cannot_be_evaluated(backend):
    from self_rewrite.data import EXCLUSION_ANSWER
    with pytest.raises(ValueError): backend.generate([Problem('x','question',EXCLUSION_ANSWER)])
