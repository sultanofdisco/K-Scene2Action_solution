import argparse
import json
from .data import prepare

def main():
    p=argparse.ArgumentParser(description="F1-T offline training preparation")
    sub=p.add_subparsers(dest="action",required=True)
    audit=sub.add_parser("prepare");audit.add_argument("--manifest",required=True);audit.add_argument("--output",required=True)
    cache=sub.add_parser("cache");cache.add_argument("--manifest",required=True)
    for name in ["cv","fit"]:
        q=sub.add_parser(name);q.add_argument("--cache",required=True);q.add_argument("--features",choices=["both","text","image"],default="both")
    for q in [cache,sub.choices['cv'],sub.choices['fit']]:
        q.add_argument("--config",default="configs/f1t-train.json");q.add_argument("--output",required=True);q.add_argument("--device",default="cpu",choices=["cpu","cuda"])
    a=p.parse_args()
    try:
        if a.action=="prepare":result=prepare(a.manifest,a.output)
        elif a.action=="cache":
            from .encoder import extract
            result=extract(a.manifest,a.config,a.output,a.device)
        else:
            from .learner import run
            result=run(a.cache,a.config,a.output,a.action,a.features,a.device)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError) as e:
        p.exit(2,f"Training preparation failed: {e}\n")

if __name__=="__main__":main()
