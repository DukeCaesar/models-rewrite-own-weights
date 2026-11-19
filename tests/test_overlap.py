import json
import pytest
from self_rewrite.backend import Problem
from self_rewrite.overlap import overlap_candidates,reviewed_exclusions


def test_math_format_variations_are_reviewable_not_auto_excluded(tmp_path):
    source=Problem('s','Compute the value of $\\frac{3}{5}+\\frac{2}{5}$.','1')
    pool=[Problem('near','Compute the value of \\dfrac{3}{5} + \\tfrac{2}{5}.','1'),
          Problem('different','Which city is the capital of France?','Paris')]
    path=tmp_path/'audit.jsonl'
    summary=overlap_candidates(pool,{'math500':[source]},path,threshold=.7)
    rows=[json.loads(x) for x in path.read_text().splitlines()]
    assert summary['candidates']==1 and rows[0]['pool_id']=='near'
    assert not rows[0]['normalized_exact'] and rows[0]['same_numbers']
    with pytest.raises(ValueError): reviewed_exclusions(path,tmp_path/'ids.json')
    rows[0].update(decision='overlap',review_note='Identical arithmetic expression; only fraction commands and spacing differ.')
    path.write_text(json.dumps(rows[0])+'\n')
    assert reviewed_exclusions(path,tmp_path/'ids.json')==['near']
