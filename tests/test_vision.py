import json
import pytest
from self_rewrite.backend import Problem
from self_rewrite.vision import image_correct,grade_image,prepare_images,BENCHMARKS
from self_rewrite.data import read_problems


def test_image_answers_require_closed_thinking():
    assert image_correct('reason</think>\\boxed{B}<|im_end|>','B','choice')
    assert not image_correct('\\boxed{B}','B','choice')
    assert image_correct('</think>The answer is (C).','C','choice')
    assert not image_correct('</think>It might be A or B.','B','choice')
    assert image_correct('</think>No, the object is absent.','no','yesno')
    assert not image_correct('</think>I do not know.','no','yesno')
    assert image_correct('</think>\\boxed{\\frac{1}{2}}','0.5','image-math')
    ref=json.dumps({'bbox':[.1,.2,.4,.6],'coordinate_scale':[1000,1000]})
    assert image_correct('</think>\\boxed{(250, 400)}',ref,'point')
    assert not image_correct('</think>(900, 400)',ref,'point')
    assert not image_correct('</think>(250,400) or (900,400)',ref,'point')
    p=Problem('x','click',ref,grading='point')
    row=grade_image(p,0,1,'loop (250,400)',100,100,'length')
    assert not row.correct and row.cut


def test_image_sampling_and_relative_paths(tmp_path):
    from PIL import Image
    image=tmp_path/'image.png';Image.new('RGB',(32,32),'white').save(image)
    sources={name:[Problem(f'{name}-{i}','Is it white?','yes',str(image),'yesno',name) for i in range(4)] for name in BENCHMARKS}
    manifest=prepare_images(sources,tmp_path/'prepared',draw_count=3,evaluate_count=2,seed=7)
    problems=read_problems(tmp_path/'prepared'/'images.jsonl')
    assert len(problems)==12
    assert all(len(x['draw_ids'])==3 for x in manifest['benchmarks'].values())
    relative=tmp_path/'relative.jsonl'
    relative.write_text(json.dumps(dict(id='one',question='White?',answer='yes',image='image.png',grading='yesno',benchmark='pope'))+'\n')
    assert read_problems(relative)[0].image==str(image)
    sources['pope'][0]=Problem('bad','click',json.dumps({'bbox':[0,0,1,1],'coordinate_scale':[0,1000]}),str(image),'point','pope')
    with pytest.raises(ValueError): prepare_images(sources,tmp_path/'bad',draw_count=3,evaluate_count=2)
