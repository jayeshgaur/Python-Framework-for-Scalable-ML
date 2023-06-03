from simpleclassifier.display import Display


def test_json_empty(capsys):
    display = Display()
    display.json({})
    captured = capsys.readouterr()
    assert captured.out == "{}\n"


def test_json_single_classifier_single_metric(capsys):
    display = Display()
    display.json({'classifier1': {'metric1': 0.9}})
    captured = capsys.readouterr()
    assert captured.out == '{\n    "classifier1": {' \
                           '\n        "metric1": 0.9\n    }\n}\n'


def test_json_multiple_classifiers_multiple_metrics(capsys):
    display = Display()
    display.json({
        'classifier1': {
            'metric1': 0.9,
            'metric2': 0.8
        },
        'classifier2': {
            'metric1': 0.85,
            'metric2': 0.75
        }
    })
    captured = capsys.readouterr()
    assert captured.out == '{\n    "classifier1": {\n        "metric1": 0.9,' \
                           '\n        "metric2": 0.8\n    },' \
                           '\n    "classifier2": {\n        "metric1": 0.85,' \
                           '\n        "metric2": 0.75\n    }\n}\n'
