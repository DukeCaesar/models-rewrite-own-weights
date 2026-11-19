from pathlib import Path
import json
import pytest
from PIL import Image
from self_rewrite.adapters import text_problem,image_problem,import_benchmark
from self_rewrite.data import read_problems


def test_text_schema_conversion():
    assert text_problem('gpqa-diamond',{'Question':'q','Correct Answer':'a'},0).question=='q'
    assert text_problem('olympiadbench',{'question':'q','final_answer':['42']},3).answer=='42'
    with pytest.raises(ValueError): text_problem('math500',{'problem':'q'},0)


def test_source_specific_image_fields(tmp_path):
    image=tmp_path/'source.png';Image.new('RGB',(200,100),'white').save(image)
    out=tmp_path/'images'
    vista=image_problem('mathvista',{'decoded_image':str(image),'question':'Which?','answer':'blue','choices':['red','blue']},0,tmp_path,out)
    assert vista.answer=='B' and 'B. blue' in vista.question
    assert image_problem('mathverse',{'problem_version':'Text Only'},1,tmp_path,out) is None
    verse=image_problem('mathverse',{'problem_version':'Vision Only','image':str(image),'question':'','query_cot':'Solve the pictured problem.','answer':'A','question_type':'multi-choice'},1,tmp_path,out)
    assert verse.question=='Solve the pictured problem.' and verse.grading=='choice'
    screen=image_problem('screenspot-v2',{'img_filename':'source.png','instruction':'button','bbox':[20,20,80,60]},2,tmp_path,out,bbox_space='pixels',bbox_format='xyxy')
    assert json.loads(screen.answer)['bbox']==[.1,.2,.4,.6]
    with pytest.raises(ValueError): image_problem('screenspot-v2',{'img_filename':'source.png','instruction':'button','bbox':[.1,.2,.4,.6]},2,tmp_path,out)
    hall=image_problem('hallusionbench',{'image':str(image),'question':'Visible?','gt_answer':'0'},3,tmp_path,out)
    assert hall.answer=='no'


def test_parquet_embedded_image_import(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    image=tmp_path/'image.png';Image.new('RGB',(32,32),'white').save(image)
    source=tmp_path/'raw.parquet'
    pq.write_table(pa.Table.from_pylist([{'question':'White?','answer':'yes','image':{'bytes':image.read_bytes(),'path':None}}]),source)
    report=import_benchmark('pope',[source],tmp_path/'converted',repository='test/repo',revision='abc')
    result=read_problems(tmp_path/'converted'/'problems.jsonl')
    assert report['imported_rows']==1 and result[0].answer=='yes'


def test_amc_year_filter_and_question_only_exclusions(tmp_path):
    source=tmp_path/'amc.json'
    source.write_text(json.dumps([{'problem':'old','url':'https://example/2022_AMC_12A_Problems'},
                                 {'problem':'current','url':'https://example/2023_AMC_12A_Problems'}]))
    report=import_benchmark('amc2023',[source],tmp_path/'amc',exclusion_only=True)
    from self_rewrite.data import EXCLUSION_ANSWER
    problems=read_problems(tmp_path/'amc'/'problems.jsonl')
    assert len(problems)==1 and problems[0].question=='current' and problems[0].answer==EXCLUSION_ANSWER
    assert report['filtered_row_indices']==[0] and report['purpose']=='exclusion-only'
    assert text_problem('olympiadbench',{'question':'prove something'},0,exclusion_only=True).answer==EXCLUSION_ANSWER


def test_screenspot_original_xywh_boxes(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(200,100),'white').save(image)
    row={'img_filename':'image.png','instruction':'button','bbox':[20,20,60,40]}
    problem=image_problem('screenspot-v2',row,0,tmp_path,tmp_path/'out',bbox_space='pixels',bbox_format='xywh')
    assert json.loads(problem.answer)['bbox']==[.1,.2,.4,.6]
    with pytest.raises(ValueError): image_problem('screenspot-v2',row,1,tmp_path,tmp_path/'out',bbox_space='pixels')


def test_duplicate_mathvista_options_preserve_all_correct_labels(tmp_path):
    from self_rewrite.vision import image_correct
    image=tmp_path/'source.png';Image.new('RGB',(32,32),'white').save(image)
    row={'decoded_image':str(image),'question':'Find CD.','answer':'18','choices':['9','12','18','18']}
    p=image_problem('mathvista',row,780,tmp_path,tmp_path/'images')
    assert p.answer=='C,D'
    assert image_correct('</think>\\boxed{C}',p.answer,p.grading)
    assert image_correct('</think>\\boxed{D}',p.answer,p.grading)
    assert not image_correct('</think>\\boxed{B}',p.answer,p.grading)


def test_raw_image_bytes_and_one_pixel_bbox_correction(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(200,100),'white').save(image)
    p=image_problem('mmstar',{'image':image.read_bytes(),'question':'Which?','answer':'A'},0,tmp_path,tmp_path/'images')
    assert p.answer=='A' and Path(p.image).is_file()
    row={'img_filename':'image.png','instruction':'button','bbox':[20,20,181,40]}
    p=image_problem('screenspot-v2',row,1,tmp_path,tmp_path/'images',bbox_space='pixels',bbox_format='xywh')
    ref=json.loads(p.answer)
    assert ref['bbox']==[.1,.2,1.,.6] and ref['original_bbox_pixels']==[20,20,201,60]
    row['bbox']=[20,20,182,40]
    with pytest.raises(ValueError,match='more than one pixel'):
        image_problem('screenspot-v2',row,2,tmp_path,tmp_path/'images',bbox_space='pixels',bbox_format='xywh')


def test_mathverse_value_and_multiple_required_choices(tmp_path):
    from self_rewrite.vision import image_correct
    image=tmp_path/'source.png';Image.new('RGB',(32,32),'white').save(image)
    row={'problem_version':'Vision Only','image':str(image),'question':'','query_cot':'Read image.','answer':'False','question_type':'multi-choice','question_for_eval':'Choice:\nA. True\nB. False'}
    p=image_problem('mathverse',row,0,tmp_path,tmp_path/'images')
    assert p.answer=='B' and p.question=='Read image.'
    row['answer']='C\nD'
    p=image_problem('mathverse',row,1,tmp_path,tmp_path/'images')
    assert p.grading=='choice-set' and p.answer=='C,D'
    assert image_correct('</think>Answer: D and C',p.answer,p.grading)
    assert not image_correct('</think>Answer: C',p.answer,p.grading)
    assert not image_correct('</think>Answer: C,D,A',p.answer,p.grading)


def test_repeated_source_images_share_lossless_file(tmp_path):
    image=tmp_path/'image.png';Image.new('RGB',(32,32),'white').save(image)
    source=tmp_path/'source.json'
    source.write_text(json.dumps([{'question':q,'answer':'yes','image':str(image)} for q in ['White?','Bright?']]))
    report=import_benchmark('pope',[source],tmp_path/'converted')
    rows=read_problems(tmp_path/'converted'/'problems.jsonl')
    assert report['unique_images']==1 and len(rows)==2
    assert rows[0].id!=rows[1].id and rows[0].image==rows[1].image
    assert Image.open(rows[0].image).tobytes()==Image.open(image).tobytes()


def test_image_archive_extraction_rejects_escape_and_changed_file(tmp_path):
    import runpy
    import zipfile
    extract=runpy.run_path(str(Path(__file__).parents[1]/'scripts'/'import_images.py'))['extract_images']
    archive=tmp_path/'images.zip';root=tmp_path/'extracted'
    with zipfile.ZipFile(archive,'w') as z: z.writestr('nested/image.bin',b'123')
    extract(archive,root);extract(archive,root)
    (root/'nested'/'image.bin').write_bytes(b'321')
    with pytest.raises(ValueError,match='differs'): extract(archive,root)
    with zipfile.ZipFile(archive,'w') as z: z.writestr('../outside.bin',b'123')
    with pytest.raises(ValueError,match='escapes'): extract(archive,root)
    assert not (tmp_path/'outside.bin').exists()
