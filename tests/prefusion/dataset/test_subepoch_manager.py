import pytest

from prefusion.dataset.subepoch_manager import SubEpochManager, EndOfAllSubEpochs
from prefusion.dataset.utils import build_subepoch_manager


def test_subepoch_manager_drop_last_false_false():
    mgr = SubEpochManager(2, drop_last_group_batch=False, drop_last_subepoch=False)
    mgr.set_batch_size(3)
    mgr.init(13)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    assert mgr.cur_subepoch_idx == 0
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 1
    assert mgr.translate_index(0) == 4
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_drop_last_false_true():
    mgr = SubEpochManager(2, drop_last_group_batch=False, drop_last_subepoch=True)
    mgr.set_batch_size(3)
    mgr.init(13)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    assert mgr.cur_subepoch_idx == 0
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_drop_last_true_false():
    mgr = SubEpochManager(2, drop_last_group_batch=True, drop_last_subepoch=False)
    mgr.set_batch_size(3)
    mgr.init(16)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 1
    assert mgr.translate_index(0) == 4
    assert mgr.translate_index(1) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_drop_last_true_true():
    mgr = SubEpochManager(2, drop_last_group_batch=True, drop_last_subepoch=True)
    mgr.set_batch_size(3)
    mgr.init(16)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    with pytest.raises(IndexError):
        _ = mgr.translate_index(3)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_reset():
    mgr = SubEpochManager(2, drop_last_group_batch=False, drop_last_subepoch=False)
    mgr.set_batch_size(5)
    mgr.init(16)
    assert mgr.num_total_group_batches == 4
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(0) == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.cur_subepoch_idx == 1
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()
    mgr.reset(13)
    assert mgr.num_total_group_batches == 3
    assert mgr.num_subepochs == 2
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.cur_subepoch_idx == 0
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    mgr.to_next_sub_epoch()
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 1
    assert mgr.cur_subepoch_idx == 1
    assert mgr.translate_index(0) == 2
    assert mgr.translate_index(1) == 1
    with pytest.raises(IndexError):
        _ = mgr.translate_index(2)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_translate_index():
    mgr = SubEpochManager(4, drop_last_group_batch=False, drop_last_subepoch=False)
    mgr.set_batch_size(2)
    mgr.init(19)
    assert mgr.num_total_group_batches == 10
    assert mgr.num_subepochs == 3
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 4
    assert mgr.translate_index(0) == 0
    assert mgr.translate_index(1) == 1
    assert mgr.translate_index(2) == 2
    assert mgr.translate_index(3) == 3
    with pytest.raises(IndexError):
        _ = mgr.translate_index(4)
    mgr.to_next_sub_epoch()
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 4
    assert mgr.translate_index(0) == 4
    assert mgr.translate_index(1) == 5
    assert mgr.translate_index(2) == 6
    assert mgr.translate_index(3) == 7
    with pytest.raises(IndexError):
        _ = mgr.translate_index(4)
    mgr.to_next_sub_epoch()
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.translate_index(0) == 8
    assert mgr.translate_index(1) == 9
    assert mgr.translate_index(2) == 2
    assert mgr.translate_index(3) == 7
    with pytest.raises(IndexError):
        _ = mgr.translate_index(4)
    with pytest.raises(EndOfAllSubEpochs):
        mgr.to_next_sub_epoch()


def test_subepoch_manager_visited():
    mgr = SubEpochManager(2, drop_last_group_batch=True, drop_last_subepoch=True, debug_mode=True)
    mgr.set_batch_size(3)
    mgr.init(16)
    assert mgr.num_total_group_batches == 5
    assert mgr.num_subepochs == 2
    assert mgr._get_num_group_batches_available_to_visit() == 6
    assert mgr.get_actual_num_group_batches_in_cur_subepoch() == 2
    assert mgr.visited.todict() == {}
    assert mgr.translate_index(0) == 0
    assert set(mgr.visited.todict()) == {0}
    assert mgr.translate_index(1) == 1
    assert set(mgr.visited.todict()) == {0, 1}
    mgr.to_next_sub_epoch()
    assert mgr.translate_index(0) == 2
    assert mgr._get_num_group_batches_available_to_visit() == 6
    assert set(mgr.visited.todict()) == {0, 1, 2}
    assert mgr.translate_index(1) == 3
    assert set(mgr.visited.todict()) == {0, 1, 2, 3}
    
    with pytest.warns(UserWarning) as warning_info:
        mgr.reset(18)
    assert str(warning_info[0].message) == "Some group batches are not visited! (group_batch_index: {4, 5})"

    assert mgr.num_total_group_batches == 6
    assert mgr.num_subepochs == 3
    assert mgr._get_num_group_batches_available_to_visit() == 6
    assert mgr.visited.todict() == {}
    assert mgr.translate_index(0) == 0
    assert set(mgr.visited.todict()) == {0}
    assert mgr.translate_index(1) == 1
    assert set(mgr.visited.todict()) == {0, 1}
    mgr.to_next_sub_epoch()
    assert mgr.translate_index(1) == 3
    assert set(mgr.visited.todict()) == {0, 1, 3}
    mgr.to_next_sub_epoch()
    assert mgr.translate_index(0) == 4
    assert set(mgr.visited.todict()) == {0, 1, 3, 4}
    with pytest.warns(UserWarning) as warning_info:
        mgr.reset(6)
    assert str(warning_info[0].message) == "Some group batches are not visited! (group_batch_index: {2, 5})"
