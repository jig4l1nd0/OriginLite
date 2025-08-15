import pandas as pd
from originlite.core.data_model import DataModel

def test_operation_log_add_column():
    dm = DataModel(pd.DataFrame({'x':[1,2,3]}))
    dm.add_column('y', dm.df['x']*2)
    assert any(r['op']=='add_column' for r in dm.operations)


def test_serialization_roundtrip():
    dm = DataModel(pd.DataFrame({'a':[1,2]}))
    dm.set_column_meta('a', unit='m')
    d = dm.to_dict()
    dm2 = DataModel.from_dict(d)
    assert dm2.columns_meta['a'].unit == 'm'
