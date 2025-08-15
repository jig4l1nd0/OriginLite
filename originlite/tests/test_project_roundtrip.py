import json
import pandas as pd
from originlite.data import DataModel
from originlite.core.replay import replay_operations


def create_sample_dm():
    df = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [10, 11, 13, 16, 20],
        'cat': ['a', 'b', 'a', 'b', 'a']
    })
    dm = DataModel(df.copy())
    dm._original_df = df.copy()
    # Simulate operations (e.g., formula creating new column and moving avg)
    dm.apply_operation('formula', formula='y*2', out_col='y2')
    dm.apply_operation(
        'moving_average', column='y', window=2, out_col='y_ma'
    )
    return dm


def test_project_roundtrip_with_operations(tmp_path):
    dm = create_sample_dm()
    # Build minimal chart/template config including operations
    template = {
        'chart_type': 'scatter',
        'x': 'x',
        'y': 'y',
        'data_model_operations': dm.operations,
        'figure_style': {},
        'custom_color_map': {}
    }
    # Simulate saving manifest (without actual .olite packaging for simplicity)
    manifest_path = tmp_path / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump({'chart_config': template}, f)
    # Reload and replay operations on new DataModel
    new_dm = DataModel(dm._original_df.copy())
    with open(manifest_path) as f:
        loaded = json.load(f)
    ops = loaded['chart_config'].get('data_model_operations', [])
    rebased_df = replay_operations(new_dm.df, ops)
    new_dm.df = rebased_df
    # After replay, new_dm.df should have the derived columns
    assert 'y2' in new_dm.df.columns
    assert 'y_ma' in new_dm.df.columns
    # Validate values for one row
    assert new_dm.df.loc[1, 'y2'] == dm.df.loc[1, 'y2']
    assert new_dm.df.loc[2, 'y_ma'] == dm.df.loc[2, 'y_ma']
