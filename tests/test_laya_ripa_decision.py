import runpy
from types import SimpleNamespace
import torch

h=runpy.run_path('scripts/train_laya_ripa_decision.py')

def toy():
    torch.manual_seed(7)
    model=SimpleNamespace(type_emb=torch.nn.Embedding(3,8),head=torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(8,2,32,dropout=0,batch_first=True,norm_first=True),2,enable_nested_tensor=False),scorer=torch.nn.Linear(8,1))
    for p in model.type_emb.parameters():p.requires_grad_(False)
    items=[{'h':torch.randn(n,8),'markers':[0,1,2],'qtype':0} for n in [4,7]]
    return model,items

def test_rlcd_updates_head_and_scorer_but_not_frozen_input():
    model,items=toy();logits=h['decision_forward'](model,items)
    loss,rl,ce=h['rlcd_loss'](logits,torch.tensor([0,1]),torch.zeros(2,dtype=torch.long),.4)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(rl) and ce>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.scorer.parameters())
    assert all(p.grad is None for p in model.type_emb.parameters())
    assert all(i['h'].grad is None for i in items)

def test_padding_does_not_change_selected_option_logits():
    model,items=toy();model.head.eval();model.scorer.eval()
    with torch.no_grad():
        batched=h['decision_forward'](model,items)
        separate=torch.cat([h['decision_forward'](model,[i]) for i in items])
    assert torch.allclose(batched,separate,atol=2e-6)

def test_rlcd_gradient_is_invariant_to_common_logit_offset():
    x=torch.tensor([[.1,.2,.3],[.4,.2,-.1]],requires_grad=True)
    y=(x.detach()+10).requires_grad_();labels=torch.tensor([0,2]);q=torch.zeros(2,dtype=torch.long)
    torch.manual_seed(4);a=h['rlcd_loss'](x,labels,q,.2)[0];a.backward()
    torch.manual_seed(4);b=h['rlcd_loss'](y,labels,q,.2)[0];b.backward()
    assert torch.allclose(x.grad,y.grad,atol=2e-4)
    assert torch.allclose(x.grad.sum(-1),torch.zeros(2),atol=2e-5)
