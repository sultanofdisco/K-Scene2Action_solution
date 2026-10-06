import hashlib
import json
from pathlib import Path
import pytest

PIL=pytest.importorskip('PIL')
from PIL import Image
from f1.training.data import load_manifest, make_folds, digest
from f1.training.encoder import prepare_image, require_token_budget, load_cache

@pytest.fixture
def dataset(tmp_path):
    rows=[]
    for group in range(4):
        path=tmp_path/f'{group}.png';Image.new('RGB',(12,8),(30*group,20,150)).save(path)
        for label in (0,1):
            command=f'Group {group}, label {label}'
            rows.append(dict(sample_id=f'{group}-{label}',pair_id=f'p{group}',group_id=f'g{group}',
                command=command,command_sha256=hashlib.sha256(command.encode()).hexdigest(),label=label,
                image_path=path.name,image_sha256=digest(path),image_width=12,image_height=8,
                language='en',source='RoboJailBench/DROID',review_status='SELECTED_AGENT_REVIEWED'))
    path=tmp_path/'selected.jsonl'
    def save():path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    save();return path,rows,save

def test_folds_preserve_pairs_and_hold_out_every_group_once(dataset):
    path,rows,_=dataset;loaded=load_manifest(path);folds=make_folds(loaded)
    seen=[]
    for fold in folds:
        assert set(fold['train_ids']).isdisjoint(fold['validation_ids'])
        tr={r['group_id'] for r in rows if r['sample_id'] in fold['train_ids']}
        va={r['group_id'] for r in rows if r['sample_id'] in fold['validation_ids']}
        assert tr.isdisjoint(va);seen.extend(fold['validation_ids'])
    assert sorted(seen)==sorted(r['sample_id'] for r in rows)

@pytest.mark.parametrize('change', ['excluded','held','hash','pair','path','repeat_group','label_bool'])
def test_invalid_data_rejected(dataset,change):
    path,rows,save=dataset
    if change=='excluded':rows[0]['source']='RoboJailBench/Permission_Related_Offense'
    elif change=='held':rows[0]['review_status']='HELD'
    elif change=='hash':rows[0]['command']+=' modified'
    elif change=='pair':rows.pop()
    elif change=='path':rows[0]['image_path']='../escape.png'
    elif change=='repeat_group':
        rows[2]['command']=rows[0]['command'];rows[2]['command_sha256']=rows[0]['command_sha256']
    else:rows[0]['label']=False
    save()
    with pytest.raises(ValueError):load_manifest(path)

def test_panorama_preserves_both_edges():
    im=Image.new('RGB',(900,100),'white')
    im.paste((255,0,0),(0,0,90,100));im.paste((0,255,0),(810,0,900,100))
    result=prepare_image(im)
    assert result.size==(224,224)
    assert result.getpixel((3,112))[0]>240
    assert result.getpixel((220,112))[1]>240

def test_token_budget_counts_special_tokens_and_never_truncates():
    def tokenizer(commands,**kwargs):
        assert kwargs['add_special_tokens'] and kwargs['truncation'] is False
        return {'input_ids':[[1]*(int(c)+2) for c in commands]}
    assert require_token_budget(tokenizer,['126'])==[128]
    with pytest.raises(ValueError,match='F1T_TEXT_TOO_LONG'):require_token_budget(tokenizer,['127'])

def test_cache_tampering_and_training_smoke(dataset,tmp_path):
    np=pytest.importorskip('numpy');pytest.importorskip('torch')
    from f1.training.learner import run
    from f1.training.data import write_json
    path,rows,_=dataset
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/f1t-train.json').read_text())
    config.update(epochs=2,batch_size=4)
    config_path=tmp_path/'config.json';write_json(config_path,config)
    cache=tmp_path/'cache';cache.mkdir()
    rng=np.random.default_rng(3);x=rng.normal(size=(len(rows),1024)).astype('float32')
    for start in [0,512]:x[:,start:start+512]/=np.linalg.norm(x[:,start:start+512],axis=1,keepdims=True)
    np.save(cache/'features.npy',x,allow_pickle=False)
    meta=dict(format_version=1,complete=True,rows=rows,config=config,encoder_lock={'test':'synthetic embeddings only'},manifest_sha256=digest(path),features_sha256=digest(cache/'features.npy'))
    write_json(cache/'metadata.json',meta)
    cv=run(cache,config_path,tmp_path/'cv','cv')
    assert cv['independent_test'] is False and len(cv['folds'])==4
    fit=run(cache,config_path,tmp_path/'fit','fit')
    assert fit['deployment_ready'] is False and fit['operational_thresholds'] is None
    from safetensors.torch import load_file
    assert all(t.isfinite().all() for t in load_file(str(tmp_path/'fit/model.safetensors')).values())
    with (cache/'features.npy').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):load_cache(cache)
