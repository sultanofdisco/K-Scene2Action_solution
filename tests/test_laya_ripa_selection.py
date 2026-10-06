import runpy

choose = runpy.run_path('scripts/train_laya_ripa.py')['choose_key']


def result(recall=1., leak=0., clear=0., ambiguous=0):
    return {'metrics': {'attack_recall': recall, 'attack_clear_rate': leak,
                        'benign_clear_rate': clear},
            'ambiguous_correct': ambiguous, 'question_accuracy': .5, 'loss': 1.}


def test_ripa_volume_cannot_hide_legacy_attack_regression():
    assert choose(result(), result()) > choose(result(.8, clear=1), result(clear=1))


def test_attack_unknown_is_not_counted_as_detection():
    assert choose(result(), result()) > choose(result(), result(.9, clear=1))


def test_equal_recall_prefers_less_clear_leakage():
    assert choose(result(.8), result(.8)) > choose(result(.8, leak=.1), result(.8, clear=1))


def test_normal_utility_is_macro_averaged_between_validation_groups():
    assert choose(result(clear=.6), result(clear=.6)) > choose(result(clear=.1), result(clear=1))
