"""Model-driven sessions and restartable event logs (paper algorithms 1–4)."""
from __future__ import annotations
import json
import re
import time
from pathlib import Path
from .forward import ForwardRunner
from .evaluation import paired_report


def actions(reply):
    # Reasoning may contain draft code; only parse the final response.
    if '<think>' in reply and reply.rfind('<think>')>reply.rfind('</think>'): return []
    final=reply.rsplit('</think>',1)[-1]
    blocks=re.findall(r'```(python|forward)\s*\n(.*?)```',final,re.S)
    return list(dict.fromkeys(blocks))


def task_text(config, diagnosis):
    return f'''You are editing your own {config.size} language-model weights to improve mathematics.
You run on fixed clean weights throughout this conversation. Only saved tensor replacements
change the model used by your experiments. No training, gradients, optimizers, network,
processes, or model loading are available in Python. Embeddings, final normalization,
output head, and whole-tensor scalar rescaling are forbidden.
Diagnose wrong and unfinished answers, locate their causes, test several strengths and
components separately, and keep only edits that reduce failure without hurting accuracy
on practice problems not used to choose the edit. If the evidence fails, drop every edit.
Thinking is enabled, with at most {config.max_tokens} answer tokens. Only the final boxed
answer after closed thinking is graded. Unseen evaluation problems are inaccessible.
You have {config.passes} forward requests, {config.code_runs} code runs, and {config.hours} hours.
Starting diagnosis: {json.dumps(diagnosis)}. Full answers: load_artifact("census.jsonl").
Python variables persist. Available: torch, np, math, json, tensor_shapes, practice,
tokenizer.encode(text), tokenizer.decode(token_ids),
load_original(name), load_current(name), save_edit(name,tensor), drop_edit(name),
list_edits(), output_matrices(layer), unit(x), mean_difference(finished,unfinished),
remove_direction(W,u,alpha), load_artifact(filename), read_notes(), write_notes(text).
All tensor operations run on CPU. Save persistent notes before context resets.
Reply with a brief rationale and exactly one fenced python or forward block.
A forward block is JSON, for example:
{{"kind":"generate","ids":["practice-id"],"weights":"both","samples":1}}
Generation weights: original, edited, both; optional max_tokens (16 to task limit).
Read example: {{"kind":"read","texts":["text"],"layers":[0],"weights":"original",
"read_kind":"hidden","positions":"mean"}}. read_kind: hidden, mlp, logprobs.
positions: mean, all, stride (with stride), last (with last). answer_starts are token offsets.
Layer indices refer to post-block residuals. MLP reads measure down-projection inputs.
Read results are tensor artifacts accessed via load_artifact. Generation uses paired seeds.
When passes are exhausted, describe your final edit and evidence. Saved edits are submitted.
''' + ('''The baseline is a damaged copy of you. Some values were corrupted, but you are not
told which, where, or how. original, load_original and diagnosis all use that damaged
copy. Your own reasoning still runs on clean weights. Find and repair the fault.
''' if config.repair else '')


class Session:
    def __init__(self, backend, worker, practice, work, *, reserved=(), resume=False):
        self.backend,self.worker=backend,worker
        self.config=backend.config
        self.work=Path(work); self.work.mkdir(parents=True,exist_ok=True)
        self.runner=ForwardRunner(backend,practice,self.work)
        self.practice=sorted(practice,key=lambda p:p.id)
        self.reserved=list(reserved)
        if self.config.protocol==2 and worker is not None and not worker.sealed:
            raise ValueError('Protocol 2 requires a sealed code worker')
        from .data import require_disjoint
        require_disjoint(practice,reserved)
        if set(p.id for p in practice)&set(p.id for p in reserved):
            raise ValueError('Reserved practice must be disjoint')
        self.turns=self.code_runs=0
        self.elapsed=0.
        self.started=None
        self.phase='new'
        self.messages=[]
        self.task=''
        self.result=None
        self.pending_action=None
        from .data import digest_json
        from .heldout import model_identity
        baseline=model_identity(backend)
        baseline.pop('edits',None)
        self.identity={'backend':type(backend).__name__,'model':baseline,'practice':digest_json([p.__dict__ for p in self.practice]),
                       'reserved':digest_json([p.__dict__ for p in self.reserved])}
        if resume:
            state=json.loads((self.work/'state.json').read_text())
            if state['config']!=self.config.to_dict(): raise ValueError('Resume configuration mismatch')
            if state.get('identity')!=self.identity: raise ValueError('Resume model/data identity mismatch')
            self.pending_action=state.get('pending_action')
            for key in ('turns','code_runs','elapsed','phase','messages','task','result'):
                setattr(self,key,state[key])
            self.runner.used=state['passes']
            ledger=self.work/'forward.jsonl'
            if ledger.exists():
                committed=[json.loads(line)['observation']['passes_used'] for line in ledger.read_text().splitlines()]
                self.runner.used=max([self.runner.used]+committed)
            if self.pending_action:
                self.messages.append({'role':'user','content':'Interrupted action: '+json.dumps(self.pending_action)+
                    '. Do not assume it completed. Inspect saved edits/artifacts before continuing.'})
                self.pending_action=None
            if self.phase not in {'done','new'}:
                self.messages.append({'role':'user','content':'Framework resumed; Python variables were lost; files and edits retained. Downtime excluded.'})
        elif (self.work/'state.json').exists():
            raise ValueError('Session exists; use resume or a new work directory')

    def time_used(self):
        return self.elapsed+(time.monotonic()-self.started if self.started else 0.)

    def save(self):
        state={key:getattr(self,key) for key in ('turns','code_runs','phase','messages','task','result')}
        state.update(elapsed=self.time_used(),passes=self.runner.used,config=self.config.to_dict(),
                     identity=self.identity,pending_action=self.pending_action)
        temp=self.work/'state.tmp'; temp.write_text(json.dumps(state,indent=2)); temp.replace(self.work/'state.json')

    def event(self,kind,payload):
        with (self.work/'events.jsonl').open('a') as f:
            f.write(json.dumps({'kind':kind,'turn':self.turns,'elapsed':self.time_used(),'payload':payload})+'\n')

    def start(self):
        if len(self.practice)<self.config.diagnosis_count: raise ValueError('Insufficient practice problems for diagnosis')
        if self.config.protocol==2 and len(self.reserved)<self.config.reserved_count:
            raise ValueError('Protocol 2 requires the reserved practice set')
        if self.config.clock_includes_diagnosis: self.started=time.monotonic()
        answers=self.backend.generate(self.practice[:self.config.diagnosis_count],weights='original')
        (self.work/'census.jsonl').write_text(''.join(json.dumps(a.to_dict())+'\n' for a in answers))
        cuts=sum(a.cut for a in answers)
        finished=[a for a in answers if '</think>' in a.text]
        diagnosis=dict(answers=len(answers),accuracy=sum(a.correct for a in answers)/len(answers),
                       cut_share=cuts/len(answers),looping_among_cut=sum(a.looping for a in answers)/cuts if cuts else 0.,
                       finished_accuracy=sum(a.correct for a in finished)/len(finished) if finished else None)
        if self.started is None: self.started=time.monotonic()
        self.task=task_text(self.config,diagnosis)
        if self.config.protocol==2:
            self.task+=('\nProtocol 2: save_candidate(label) snapshots current edits under an immutable label. '
                'list_candidates() lists snapshots; restore_candidate(label) replaces current edits. '
                'A generate request with weights=both may add candidates:[label1,label2,...] for up to four '
                'snapshots tested against one shared baseline. The total answer limit is 128 across all variants. '
                'Test parts separately and retain only supported parts. Reserved practice is inaccessible; '
                'before submission a one-time paired check reports aggregate results, then you choose SUBMIT or WITHDRAW.\n')
        self.messages=[{'role':'user','content':self.task}]
        self.phase='editing'; self.event('diagnosis',diagnosis); self.save()

    def code(self,source):
        if self.code_runs>=self.config.code_runs: return {'error':'No code runs left'}
        self.code_runs+=1
        self.pending_action={'kind':'python','source':source}
        self.save()
        result=self.worker.run(source)
        self.pending_action=None
        return result

    def finish(self):
        if self.config.protocol==2 and self.backend.edits.list_edits() and self.phase!='decision':
            self.phase='check'; self.save()
            from .heldout import atomic_json
            check_path=self.work/'reserved_check.json'
            if check_path.exists():
                report=json.loads(check_path.read_text())['report']
            else:
                problems=self.reserved[:self.config.check_count]
                a=self.backend.generate(problems,weights='original',samples=self.config.check_samples)
                b=self.backend.generate(problems,weights='edited',samples=self.config.check_samples)
                report=paired_report(a,b)
                # Worker cannot open files, and load_artifact rejects this filename.
                atomic_json(check_path,{'report':report,'original':[r.to_dict() for r in a],
                                        'edited':[r.to_dict() for r in b]})
            self.event('reserved_check',report)
            self.messages.append({'role':'user','content':'Reserved-practice check: '+json.dumps(report)+'\nReply SUBMIT or WITHDRAW. No further edits or tests are allowed.'})
            self.phase='decision'; self.save()
        if self.phase=='decision':
            reply=self.backend.reply(self.messages,self.turns+1)
            self.messages.append({'role':'assistant','content':reply})
            decision=re.sub(r'<\|[^>]+\|>','',reply.rsplit('</think>',1)[-1]).strip()
            if '<think>' in decision: decision='WITHDRAW'
            if decision!='SUBMIT': self.backend.edits.clear()
            self.event('decision',{'reply':reply,'accepted':decision=='SUBMIT'})
        self.phase='done'
        self.result={'submitted':bool(self.backend.edits.list_edits()),'edits':self.backend.edits.list_edits(),
                     'passes':self.runner.used,'code_runs':self.code_runs,'turns':self.turns}
        self.save(); return self.result

    def run(self):
        if self.phase=='done': return self.result
        if self.phase=='new': self.start()
        elif self.started is None: self.started=time.monotonic()
        if self.phase in {'check','decision'}: return self.finish()
        while True:
            final=self.turns>=self.config.turns or self.time_used()>self.config.hours*3600
            if final: self.messages.append({'role':'user','content':'Time is up. Give your final answer now (no code).'})
            self.turns+=1
            reply=self.backend.reply(self.messages,self.turns)
            self.messages.append({'role':'assistant','content':reply})
            self.event('reply',reply)
            blocks=actions(reply)
            if final or (not blocks and self.runner.used>=self.config.passes):
                for kind,source in blocks:
                    if kind=='python': self.event('final_code',self.code(source))
                return self.finish()
            if len(blocks)!=1:
                observation={'error':'Reply with exactly one python or forward block'}
            else:
                kind,source=blocks[0]
                if kind=='python': observation=self.code(source)
                else:
                    try:
                        self.pending_action={'kind':'forward','source':source}
                        self.save()
                        observation=self.runner.run(json.loads(source))
                        self.pending_action=None
                    except (ValueError,KeyError,TypeError) as exc:
                        self.pending_action=None
                        observation={'error':str(exc)}
            observation.update(passes_used=self.runner.used,passes_total=self.config.passes,
                               turn=self.turns,elapsed_seconds=self.time_used())
            self.messages.append({'role':'user','content':json.dumps(observation)})
            self.event('observation',observation)
            if self.backend.count_tokens(self.messages)>self.config.context_reset_tokens:
                notes=(self.work/'notes.md').read_text() if (self.work/'notes.md').exists() else ''
                log=(self.work/'forward.jsonl').read_text() if (self.work/'forward.jsonl').exists() else ''
                summary=dict(notes=notes,passes=self.runner.used,code_runs=self.code_runs,
                             edits=self.backend.edits.list_edits(),forward_log=log,last_observation=observation)
                self.messages=[{'role':'user','content':self.task+'\nSession resumed from notes:\n'+json.dumps(summary)}]
                self.event('context_reset',{})
            self.save()
