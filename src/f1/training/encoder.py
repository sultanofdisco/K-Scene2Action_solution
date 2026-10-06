"""Frozen multilingual text encoder and compatible original CLIP image encoder."""
from pathlib import Path
import json
import sys

from .data import digest, load_manifest, read_config, write_json

def prepare_image(image):
    from PIL import Image, ImageOps
    # Preserve every camera view in panoramic inputs. Never silently center crop.
    image=ImageOps.exif_transpose(image).convert("RGB")
    return ImageOps.pad(image,(224,224),method=Image.Resampling.BICUBIC,
                        color=(123,117,104),centering=(0.5,0.5))

def require_token_budget(tokenizer, commands, maximum=128):
    tokens=tokenizer(commands,add_special_tokens=True,padding=False,truncation=False)["input_ids"]
    lengths=[len(t) for t in tokens]
    too_long=[i for i,n in enumerate(lengths) if n>maximum]
    if too_long: raise ValueError(f"F1T_TEXT_TOO_LONG: command indices {too_long}; no truncation performed")
    return lengths

class FrozenClip:
    def __init__(self, config, device="cpu"):
        from huggingface_hub import HfApi
        from sentence_transformers import SentenceTransformer
        self.config=config
        # Resolve moving references once, then load exactly those commits.
        api=HfApi()
        self.lock={k:{"model":config[k+"_model"],
                      "revision":api.model_info(config[k+"_model"],revision=config[k+"_revision"],timeout=30).sha}
                   for k in ("text","image")}
        self.text=SentenceTransformer(self.lock["text"]["model"],revision=self.lock["text"]["revision"],device=device,trust_remote_code=False)
        self.image=SentenceTransformer(self.lock["image"]["model"],revision=self.lock["image"]["revision"],device=device,trust_remote_code=False)
        for model in (self.text,self.image):
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
        if self.text.max_seq_length!=128: raise ValueError("Unexpected text model token limit")

    def encode(self, commands, image_paths, batch_size):
        import numpy as np
        import torch
        from PIL import Image
        lengths=require_token_budget(self.text.tokenizer,commands,self.config["max_tokens"])
        with torch.inference_mode():
            text=self.text.encode(commands,batch_size=batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)
            chunks=[]
            for start in range(0,len(image_paths),batch_size):
                images=[]
                for path in image_paths[start:start+batch_size]:
                    with Image.open(path) as im:images.append(prepare_image(im))
                chunks.append(self.image.encode(images,batch_size=batch_size,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False))
            image=np.concatenate(chunks)
        for block in (text,image):
            if block.shape[1]!=512 or not np.isfinite(block).all():raise ValueError("Invalid CLIP embeddings")
        return text.astype("float32"),image.astype("float32"),lengths

def extract(manifest, config_path, output, device="cpu"):
    import numpy as np
    from importlib.metadata import version
    import torch
    rows=load_manifest(manifest);config=read_config(config_path)
    if device=="cpu":torch.set_num_threads(config["cpu_threads"])
    root=Path(manifest).resolve().parent;out=Path(output);out.mkdir(parents=True,exist_ok=False)
    print("Loading frozen CLIP encoders...",file=sys.stderr,flush=True)
    encoder=FrozenClip(config,device)
    commands=list(dict.fromkeys(r["command"] for r in rows))
    images={r["image_sha256"]:root/r["image_path"] for r in rows}
    print(f"Encoding {len(commands)} commands and {len(images)} images on {device}...",file=sys.stderr,flush=True)
    text,image,lengths=encoder.encode(commands,list(images.values()),config["batch_size"])
    text_by={s:v for s,v in zip(commands,text)};image_by=dict(zip(images,image))
    features=np.stack([np.concatenate((text_by[r["command"]],image_by[r["image_sha256"]])) for r in rows])
    np.save(out/"features.npy",features,allow_pickle=False)
    metadata={"format_version":1,"manifest_sha256":digest(manifest),"features_sha256":digest(out/"features.npy"),
              "config":config,"encoder_lock":encoder.lock,"feature_order":["text_l2_512","image_l2_512"],
              "max_observed_tokens":max(lengths),"unique_commands":len(commands),"unique_images":len(images),
              "rows":rows,"complete":True,
              "package_versions":{name:version(name) for name in ["torch","numpy","Pillow","sentence-transformers","transformers"]}}
    # Only a completed cache receives its metadata marker.
    write_json(out/"metadata.json",metadata)
    return {k:metadata[k] for k in ["manifest_sha256","encoder_lock","max_observed_tokens","unique_commands","unique_images"]}

def load_cache(path):
    import numpy as np
    from .data import EXCLUDED
    path=Path(path);meta=json.loads((path/"metadata.json").read_text(encoding="utf8"))
    if meta.get("complete") is not True or meta.get("format_version")!=1:raise ValueError("Incomplete/unknown cache")
    if meta["features_sha256"]!=digest(path/"features.npy"):raise ValueError("Cache checksum mismatch")
    x=np.load(path/"features.npy",allow_pickle=False)
    if x.shape!=(len(meta["rows"]),1024) or not np.isfinite(x).all():raise ValueError("Cache shape/nonfinite values")
    for block in (x[:,:512],x[:,512:]):
        if not np.allclose(np.linalg.norm(block,axis=1),1,atol=1e-4):raise ValueError("Embeddings must be L2 normalized separately")
    if any(type(r["label"]) is not int or r["label"] not in (0,1) or r["review_status"]!="SELECTED_AGENT_REVIEWED" or r['source'].split('/')[-1] in EXCLUDED for r in meta["rows"]):
        raise ValueError("Cache contains unselected/nonbinary rows")
    rows=meta['rows']
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate cached sample')
    from collections import defaultdict
    pairs=defaultdict(list);links=defaultdict(set)
    for r in rows:
        pairs[r['pair_id']].append(r)
        links[('image',r['image_sha256'])].add(r['group_id'])
        links[('text',' '.join(r['command'].casefold().split()))].add(r['group_id'])
    if any(len(g)!=1 for g in links.values()):raise ValueError('Cached duplicate crosses groups')
    for pair in pairs.values():
        if len(pair)!=2 or {r['label'] for r in pair}!={0,1} or len({r['group_id'] for r in pair})!=1:
            raise ValueError('Invalid cached contrast pair')
    return x,meta
