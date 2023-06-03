from simpleclassifier.display import Display

START_END_BORDER = "=====================================\n"
METRIC_BORDER = "-------------------------------------\n"


def test_empty_dump(capsys):
    display = Display()
    display.dump({})
    captured = capsys.readouterr()
    assert captured.out == START_END_BORDER + START_END_BORDER


def test_single_classifier_single_metric(capsys):
    display = Display()
    display.dump({'classifier1': {'metric1': 0.7}})
    captured = capsys.readouterr()
    assert captured.out == f"{START_END_BORDER}{METRIC_BORDER}metric1" \
                           f"\n- classifier1: 0.7\n{START_END_BORDER}"


def test_multiple_classifiers_single_metric(capsys):
    display = Display()
    display.dump({
        'classifier1': {
            'metric1': 0.7
        },
        'classifier2': {
            'metric1': 0.9
        }
    })
    captured = capsys.readouterr()
    assert captured.out in [
        f"{START_END_BORDER}{METRIC_BORDER}metric1"
        f"\n- classifier1: 0.7\n- classifier2: 0.9"
        f"\n{START_END_BORDER}", f"{START_END_BORDER}{METRIC_BORDER}metric1"
        f"\n- classifier2: 0.9\n- classifier1: 0.7"
        f"\n{START_END_BORDER}"
    ]


def test_single_classifier_multiple_metric(capsys):
    display = Display()
    display.dump({'classifier1': {'metric1': 0.7, 'metric2': 0.8}})
    captured = capsys.readouterr()
    assert captured.out in [
        f"{START_END_BORDER}{METRIC_BORDER}metric1"
        f"\n- classifier1: 0.7\n{METRIC_BORDER}metric2"
        f"\n- classifier1: 0.8\n{START_END_BORDER}",
        f"{START_END_BORDER}{METRIC_BORDER}metric2"
        f"\n- classifier1: 0.8\n{METRIC_BORDER}metric1"
        f"\n- classifier1: 0.7\n{START_END_BORDER}"
    ]


def test_multiple_classifiers_multiple_metrics(capsys):
    display = Display()
    display.dump({
        'classifier1': {
            'metric1': 0.7,
            'metric2': 0.8
        },
        'classifier2': {
            'metric1': 0.9,
            'metric2': 0.75
        }
    })
    captured = capsys.readouterr()
    assert START_END_BORDER in captured.out
    assert METRIC_BORDER in captured.out
    assert "metric1" in captured.out
    assert "metric2" in captured.out
    assert "- classifier1: 0.7" in captured.out
    assert "- classifier2: 0.9" in captured.out
    assert "- classifier1: 0.8" in captured.out
    assert "- classifier2: 0.75" in captured.out
    assert captured.out.count(START_END_BORDER) == 2
    assert captured.out.count(METRIC_BORDER) == 2


def test_classifier_with_no_metrics(capsys):
    display = Display()
    display.dump({'classifier1': {}})
    captured = capsys.readouterr()
    assert captured.out == START_END_BORDER + START_END_BORDER
