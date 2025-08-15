import pandas as pd
from originlite.core.data_model import DataModel
from originlite.project.serializer import save_project, load_project
from originlite.core.replay import replay_operations


def build_sample_dm():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 11, 13, 16, 20],
    })
    dm = DataModel(df.copy())
    dm.apply_operation('formula', formula='y*2', out_col='y2')
    dm.apply_operation('moving_average', column='y', window=2, out_col='y_ma')
    return dm


def test_project_save_load_includes_operations(tmp_path):
    dm = build_sample_dm()
    chart_cfg = {
        'chart_type': 'scatter',
        'x': 'x',
        'y': 'y',
    }
    proj_path = tmp_path / 'sample.olite'
    save_project(proj_path, dm, chart_cfg)

    loaded_dm, loaded_chart, _ = load_project(proj_path)
    # Operations persisted in both dm and chart config
    assert loaded_dm.operations
    assert loaded_chart.get('data_model_operations')
    # Replay on fresh original data reproduces derived columns
    base_df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 11, 13, 16, 20],
    })
    replayed = replay_operations(base_df, loaded_dm.operations)
    for col in ['y2', 'y_ma']:
        assert col in replayed.columns
