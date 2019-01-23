import pytest

from delira.training import Parameters
from delira.utils import LookupConfig


@pytest.mark.parametrize("fixed_model_params,variable_model_params,"
                         "fixed_training_params,variable_training_params,"
                         "valid_nested_key,valid_nested_value,"
                         "doubled_key,invalid_key",
                         [
                             (
                                 {"a": 1, "b": [1, 2], "c": {"d": 3, "e": 56}},
                                 {"f": 1, "g": {"h": {"i": {"a": 3}}}},
                                 {"j": 1, "k": 2},
                                 {},
                                 "e",
                                 56,
                                 "a",
                                 "q"
                             )
                         ])
def test_parameters(fixed_model_params, variable_model_params,
                    fixed_training_params, variable_training_params,
                    valid_nested_key, valid_nested_value,
                    doubled_key, invalid_key):

    def to_lookup_config(dictionary):
        tmp = LookupConfig()
        tmp.update(dictionary)
        return tmp
    fixed_model_params = to_lookup_config(fixed_model_params)
    variable_model_params = to_lookup_config(variable_model_params)
    fixed_training_params = to_lookup_config(fixed_training_params)
    variable_training_params = to_lookup_config(variable_training_params)

    params = Parameters(
        fixed_params={
            "model": fixed_model_params,
            "training": fixed_training_params
            },
        variable_params={
            "model": variable_model_params,
            "training": variable_training_params
            }
        )

    assert params.training_on_top == False
    assert params.variability_on_top

    assert params.fixed == to_lookup_config({
        "model": fixed_model_params,
        "training": fixed_training_params
    })

    assert params.variable == to_lookup_config({
        "model": variable_model_params,
        "training": variable_training_params
    })

    
    params = params.permute_training_on_top()

    assert params.variability_on_top == False
    assert params.training_on_top

    print(params.model.difference_config(to_lookup_config({
        "fixed": fixed_model_params,
        "variable": variable_model_params
    })))

    assert params.model == to_lookup_config({
        "fixed": fixed_model_params,
        "variable": variable_model_params
    })

    assert params.training == to_lookup_config({
        "fixed": fixed_training_params,
        "variable": variable_training_params
    })

    params_copy = params.deepcopy()
    params = params.permute_variability_on_top().permute_training_on_top()
    assert params_copy == params

    assert params.nested_get(valid_nested_key) == valid_nested_value
    
    try:
        params.nested_get(doubled_key)
        assert False
    except KeyError:
        assert True

    try:
        params.nested_get(invalid_key)
        assert False
    except KeyError:
        assert True

    assert "default" == params.nested_get(invalid_key, "default")
    assert "default" == params.nested_get(invalid_key, default="default")

if __name__ == '__main__':
    test_parameters()
