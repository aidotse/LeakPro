"""Tests for utils module."""
import os

import leakpro.synthetic_data_attacks.utils as u


def test_aux_file_path() -> None:
    """Assert results of aux_file_path function."""
    e_file_path_pre = "/LeakPro/leakpro/synthetic_data_attacks/results/"
    #Case prefix==""
    file_path = u.aux_file_path(prefix="", dataset="test")
    assert file_path[-61:] == e_file_path_pre + "res_test.json"
    #Case prefix!=""
    file_path = u.aux_file_path(prefix="testing", dataset="test")
    assert file_path[-69:] == e_file_path_pre + "res_testing_test.json"
    #Case path not None
    path = "/test"
    file_path = u.aux_file_path(prefix="", dataset="test", path=path)
    assert file_path == path + "/" + "res_test.json"
    path = "/test/"
    file_path = u.aux_file_path(prefix="", dataset="test", path=path)
    assert file_path == path + "res_test.json"

def test_save_load_res_json_file() -> None:
    """Assert results of save_res_json_file and load_res_json_file functions."""
    #Setup test variables
    res = {"0": 0, "1": 1}
    prefix = "test"
    dataset = "test_save_load_res_json_file"
    for path in [None, "/tmp", "/tmp/"]: # noqa: S108
        file_path = u.aux_file_path(prefix=prefix, dataset=dataset, path=path)
        if path is not None:
            extra = ""
            if path[-1] == "/":
                extra = "/"
            assert os.path.dirname(file_path) + extra == path
        else:
            assert os.path.dirname(file_path) + "/" == u.DEFAULT_PATH_RESULTS
        #Test save_res_json_file
        assert not os.path.exists(file_path)
        u.save_res_json_file(prefix=prefix, dataset=dataset, res=res, path=path)
        assert os.path.exists(file_path)
        #Test load_res_json_file
        new_res = u.load_res_json_file(prefix=prefix, dataset=dataset, path=path)
        assert new_res == res
        #Remove file
        os.remove(file_path)
        assert not os.path.exists(file_path)
