from gsax.problem import Problem


def test_from_dict():
    p = Problem.from_dict({"x1": (0.0, 1.0), "x2": (-1.0, 1.0)})
    assert p.names == ("x1", "x2")
    assert p.bounds == ((0.0, 1.0), (-1.0, 1.0))
    assert p.num_vars == 2


def test_frozen():
    p = Problem(names=("a",), bounds=((0.0, 1.0),))
    try:
        p.names = ("b",)
        assert False, "Should be frozen"
    except AttributeError:
        pass
