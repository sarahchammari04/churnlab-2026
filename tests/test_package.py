import churnlab
from churnlab.data import FEATURES, TARGET


def test_version():
    assert churnlab.__version__ == "0.1.0"


def test_target_not_in_features():
    assert TARGET not in FEATURES
    assert "customerID" not in FEATURES
