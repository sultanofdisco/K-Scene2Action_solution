import runpy

key=runpy.run_path('scripts/retrain_laya_full.py')['selection_key']

def metrics(detected=9,cleared=0,benign=0,ambiguous=0):
    return {'attack_n':9,'attack_detect_count':detected,'attack_clear_count':cleared,
            'benign_clear_count':benign,'ambiguous_correct':ambiguous}

def test_leaking_candidate_cannot_replace_all_attack_detection():
    assert key(metrics(),20,1.5)>key(metrics(8,1,11,2),85,0.1)

def test_benign_utility_beats_lower_loss_when_attacks_retained():
    assert key(metrics(benign=1),30,1.2)>key(metrics(),70,0.1)

def test_unknown_attack_is_not_attack_detection():
    assert key(metrics(),20,1.5)>key(metrics(8,0,11,2),85,0.1)

def test_ambiguous_resolution_breaks_utility_tie():
    assert key(metrics(benign=2,ambiguous=1),50,1.0)>key(metrics(benign=2),60,0.1)
