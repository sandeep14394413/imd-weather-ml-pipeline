from src.train_baseline import make_demo_training_data, train_baseline


def test_baseline_model_contains_lookup():
    frame = make_demo_training_data()
    model = train_baseline(frame)
    assert model["type"] == "seasonal_climatology"
    assert ("Delhi", 1) in model["lookup"]
