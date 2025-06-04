from src.pipelines.main import run_pipeline


def test_run_pipeline(tmp_path):
    data_path = tmp_path / 'data.csv'
    data_path.write_text('feature1,feature2,target\n1,10,0\n2,20,1\n3,30,0\n4,40,1')
    config = {
        'data_path': str(data_path),
        'target_column': 'target',
        'model_path': str(tmp_path / 'model.joblib'),
        'param_grid': {
            'max_depth': [3],
            'subsample': [1.0]
        },
        'cv': 2
    }
    run_pipeline(config)
    assert (tmp_path / 'model.joblib').exists()
