import LOPART
import pandas as pd
import pytest

data = [1.1, 2.2, 5.5, 6.6]


def test_LOPARTNoLabels():
    # Empty Labels
    labels = pd.DataFrame(columns=['start', 'end', 'change'])
    penalty = 10
    output = LOPART.runLOPART(data, labels, penalty)
    assert len(output['segments'].index) == 2

    firstSegment = output['segments'].iloc[0]

    assert firstSegment['start'] == 1

    assert firstSegment['end'] == 2

    # Height is actually 1.6500000000000001 so I'm checking that it's close enough
    assert abs(firstSegment['height'] - 1.65) < 0.0001

    print('noLabelOutput', output['segments'])


def test_LOPARTPositiveLabel():
    # Empty Labels
    labels = pd.DataFrame({'start': [1], 'end': [len(data)], 'change': [1]})
    penalty = 10
    output = LOPART.runLOPART(data, labels, penalty)

    assert len(output['segments'].index) == 2


def test_LOPARTNegativeLabel():
    labels = pd.DataFrame({'start': [1], 'end': [len(data)], 'change': [0]})
    penalty = 10
    output = LOPART.runLOPART(data, labels, penalty)

    assert len(output['segments'].index) == 1

def test_slimLOPART():
    labels = pd.DataFrame({'start': [1], 'end': [len(data)], 'change': [0]})
    penalty = 10
    output = LOPART.runSlimLOPART(data, labels, penalty)

    assert len(output.index) == 1