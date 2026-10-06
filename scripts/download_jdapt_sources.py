"""Acquire public source data for defensive F1 research; never run upstream code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

from huggingface_hub import HfApi, snapshot_download, hf_hub_download


def digest(path):
    h = sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def get_file(url, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():
        with urllib.request.urlopen(url,timeout=90) as r, target.with_suffix(target.suffix+'.part').open('wb') as w:
            shutil.copyfileobj(r,w)
        target.with_suffix(target.suffix+'.part').replace(target)
    return {'url':url,'path':str(target),'bytes':target.stat().st_size,'sha256':digest(target)}


def download_jb(root):
    repo='JailbreakV-28K/JailBreakV-28k'; api=HfApi();info=api.dataset_info(repo,revision='f949ca582fff13d396ac8fce59596afafb2b78d3',files_metadata=True)
    folder=root/'raw/jailbreakv'
    patterns=['README.md','JailBreakV_28K/*.csv','JailBreakV_28K/**/*.png','JailBreakV_28K/**/*.jpg','JailBreakV_28K/**/*.jpeg']
    print('Downloading public JailBreakV text and images',info.sha,flush=True)
    snapshot_download(repo,repo_type='dataset',revision=info.sha,local_dir=folder,allow_patterns=patterns,max_workers=6)
    files=[]
    for entry in info.siblings:
        p=folder/entry.rfilename
        if not p.is_file():continue
        h=digest(p)
        if entry.lfs and entry.lfs.sha256 != h:raise ValueError('LFS hash mismatch: '+entry.rfilename)
        files.append({'path':entry.rfilename,'bytes':p.stat().st_size,'sha256':h})
    result={'repo':repo,'revision':info.sha,'files':files,'license_card':str(info.card_data.get('license'))}
    save(root/'provenance/jailbreakv-lock.json',result)
    print('JailBreakV files verified:',len(files),flush=True)
    return result


def download_daquar(root):
    repo='sudiphalder/daquar'; info=HfApi().dataset_info(repo,revision='a558b3927dcf7136b06e2789eaea648a788ca9c0',files_metadata=True)
    folder=root/'raw/daquar_hf';print('Downloading DAQUAR image mirror',info.sha,flush=True)
    p=Path(hf_hub_download(repo,'dataset.zip',repo_type='dataset',revision=info.sha,local_dir=folder))
    entry=next(e for e in info.siblings if e.rfilename=='dataset.zip');h=digest(p)
    if entry.lfs and h != entry.lfs.sha256:raise ValueError('DAQUAR archive hash mismatch')
    output=root/'raw/daquar_hf/extracted';output.mkdir(exist_ok=True)
    members=[]
    with zipfile.ZipFile(p) as z:
        for item in z.infolist():
            relative=PurePosixPath(item.filename)
            if item.is_dir() or relative.suffix.lower() not in {'.png','.jpg','.jpeg','.csv','.txt','.md','.json'}:continue
            dest=(output/Path(*relative.parts)).resolve()
            if relative.is_absolute() or '..' in relative.parts or not dest.is_relative_to(output.resolve()):raise ValueError('Unsafe archive path')
            if item.file_size > 64*1024*1024:raise ValueError('Unexpected oversized member')
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                with z.open(item) as r,dest.open('wb') as w:shutil.copyfileobj(r,w)
            members.append({'path':str(dest.relative_to(root)),'bytes':dest.stat().st_size,'sha256':digest(dest)})
    result={'repo':repo,'revision':info.sha,'archive_sha256':h,'archive_bytes':p.stat().st_size,'members':members,
            'role':'Unofficial HF mirror; compare question counts and image IDs with official DAQUAR annotations'}
    save(root/'provenance/daquar-mirror-lock.json',result)
    print('DAQUAR extracted files:',len(members),flush=True)
    return result


def download_official(root):
    urls={
      'qa.894.raw.train.txt':'https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.train.txt',
      'qa.894.raw.test.txt':'https://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/qa.894.raw.test.txt',
      'source-page.html':'https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/visual-turing-challenge',
    }
    result=[]
    for name,url in urls.items():
        try:result.append(get_file(url,root/'raw/daquar_official'/name))
        except Exception as e:result.append({'url':url,'error':type(e).__name__+': '+str(e)})
    save(root/'provenance/daquar-official-sources.json',result)
    if any('error' in item for item in result):raise RuntimeError('Some official DAQUAR downloads failed; see provenance')
    return result


def download_processed_github(root):
    repo='tezansahu/VQA-With-Multimodal-Transformers'
    revision='e737e61ffa1553b6593579b7c296fbd724c782f8'
    names=['LICENSE','README.md']
    names += ['dataset/'+name for name in ['all_qa_pairs.txt','answer_space.txt','data.csv','data_eval.csv','data_train.csv','test_images_list.txt','train_images_list.txt']]
    names += ['src/'+name+'.py' for name in ['data_collator','evaluate','inference','load_data','main','model','process_dataset','train','utils']]
    entries=[]
    for name in names:
        item=get_file(f'https://raw.githubusercontent.com/{repo}/{revision}/{name}',root/'raw/daquar_processed_github'/name)
        item['path']=str(Path(item['path']).relative_to(root))
        entries.append(item)
    result={'repo':repo,'revision':revision,'files':entries}
    save(root/'provenance/daquar-processed-github-lock.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    save(root/'provenance/acquisition.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'purpose':'Public data acquisition and audit only. No training or attack execution.'})
    with ThreadPoolExecutor(max_workers=3) as pool:
        tasks={name:pool.submit(fn,root) for name,fn in [('jailbreakv',download_jb),('daquar',download_daquar),('official',download_official),('processed_github',download_processed_github)]}
        status={}
        for name,task in tasks.items():
            try:task.result();status[name]='complete'
            except Exception as e:status[name]=type(e).__name__+': '+str(e)
            save(root/'provenance/acquisition-status.json',status)
    print(json.dumps(status),flush=True)


if __name__=='__main__':main()
