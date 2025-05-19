import pytest

from prefusion.dataset.index_info import IndexInfo, establish_linkings, establish_group_linkings

def test_index_info_basic():
    ii = IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'), g_next=IndexInfo.from_str('Scn/128'))
    assert ii.as_dict() == {'scene_id': 'Scn', 'frame_id': '127', 'g_prev': {'scene_id': 'Scn', 'frame_id': '126'}, 'g_next': {'scene_id': 'Scn', 'frame_id': '128'}, 'next': None, 'prev': None}
    assert ii.g_prev.as_dict() == {'scene_id': 'Scn', 'frame_id': '126', 'g_prev': None, 'g_next': {'scene_id': 'Scn', 'frame_id': '127'}, 'next': None, 'prev': None}
    assert ii.g_next.as_dict() == {'scene_id': 'Scn', 'frame_id': '128', 'g_prev': {'scene_id': 'Scn', 'frame_id': '127'}, 'g_next': None, 'next': None, 'prev': None}

def test_index_info_modify():
    ii = IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'), g_next=IndexInfo.from_str('Scn/128'))
    assert ii.as_dict() == {'scene_id': 'Scn', 'frame_id': '127', 'g_prev': {'scene_id': 'Scn', 'frame_id': '126'}, 'g_next': {'scene_id': 'Scn', 'frame_id': '128'}, 'next': None, 'prev': None}
    ii.frame_id = '888'
    assert ii.g_prev.as_dict() == {'scene_id': 'Scn', 'frame_id': '126', 'g_prev': None, 'g_next': {'scene_id': 'Scn', 'frame_id': '888'}, 'next': None, 'prev': None}
    assert ii.g_next.as_dict() == {'scene_id': 'Scn', 'frame_id': '128', 'g_prev': {'scene_id': 'Scn', 'frame_id': '888'}, 'g_next': None, 'next': None, 'prev': None}


def test_index_info_eq():
    assert IndexInfo('Scn', '127') == IndexInfo('Scn', '127')
    assert IndexInfo('Scn', '127') != IndexInfo('Scn', '333')
    assert IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126')) == IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'))
    assert IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126')) == IndexInfo('Scn', '126', g_next=IndexInfo('Scn', '127')).g_next
    assert IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126')) != IndexInfo('Scn', '126', g_next=IndexInfo('Scn', '127'))
    assert IndexInfo('Scn', '127', g_next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '127', g_next=IndexInfo('Scn', '128'))
    assert IndexInfo('Scn', '127', g_next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '128', g_prev=IndexInfo('Scn', '127')).g_prev
    assert IndexInfo('Scn', '127', g_next=IndexInfo('Scn', '128')) != IndexInfo('Scn', '128', g_prev=IndexInfo('Scn', '127'))
    assert IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'), g_next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'), g_next=IndexInfo('Scn', '128'))
    assert IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'), g_next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '128', g_prev=IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'))).g_prev
    assert IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126'), g_next=IndexInfo('Scn', '128')) != IndexInfo('Scn', '128', g_prev=IndexInfo('Scn', '127', g_prev=IndexInfo('Scn', '126')))


def test_index_info_initialization():
    node = IndexInfo("scene1", "frame23")
    assert node.scene_id == "scene1"
    assert node.frame_id == "frame23"
    assert node.prev is None
    assert node.next is None
    assert node.g_prev is None
    assert node.g_next is None
def test_index_info_linking_prev_next():
    node_a = IndexInfo("s1", "f1")
    node_b = IndexInfo("s1", "f2", prev=node_a)
    assert node_b.prev == node_a
    assert node_a.next == node_b
def test_index_info_linking_next_in_constructor():
    node_a = IndexInfo("s1", "f1")
    node_b = IndexInfo("s1", "f2", next=node_a)
    assert node_a.prev == node_b
    assert node_b.next == node_a
def test_index_info_group_linking():
    node_a = IndexInfo("s1", "f1")
    node_b = IndexInfo("s1", "f2", g_prev=node_a)
    assert node_b.g_prev == node_a
    assert node_a.g_next == node_b
def test_scene_frame_id_property():
    node = IndexInfo("scene_0", "frame100")
    assert node.scene_frame_id == "scene_0/frame100"
def test_index_info_equality():
    node1 = IndexInfo("s1", "f1")
    node2 = IndexInfo("s1", "f1")
    assert node1 == node2
    node3 = IndexInfo("s1", "f2")
    assert node1 != node3
    node4 = IndexInfo("s1", "f1", prev=node3)
    assert node1 != node4
def test_as_dict_method():
    node = IndexInfo("s5", "f99")
    expected_dict = {
        "scene_id": "s5",
        "frame_id": "f99",
        "prev": None,
        "next": None,
        "g_prev": None,
        "g_next": None,
    }
    assert node.as_dict() == expected_dict
def test_from_str_method():
    index_str = "scene_a/frame_001"
    node = IndexInfo.from_str(index_str)
    assert node.scene_id == "scene_a"
    assert node.frame_id == "frame_001"
def test_from_str_with_custom_separator():
    index_str = "scene_x#frame_45"
    node = IndexInfo.from_str(index_str, sep="#")
    assert node.scene_id == "scene_x"
    assert node.frame_id == "frame_45"
def test_from_str_invalid_input():
    with pytest.raises(ValueError):
        IndexInfo.from_str("invalid_scene_frame_id")
# Test cases for establish_linkings and establish_group_linkings
def test_establish_linkings_prev_pointers():
    node1 = IndexInfo("s1", "f1")
    node2 = IndexInfo("s1", "f2")
    nodes = [node1, node2]
    establish_linkings(nodes)
    
    assert node1.prev is None
    assert node1.next is node2
    assert node2.prev is node1
    assert node2.next is None
def test_establish_linkings_with_single_node():
    node = IndexInfo("s1", "f0")
    result = establish_linkings([node])
    assert result == [node]
    assert node.prev is None
    assert node.next is None
def test_establish_group_linkings():
    node1 = IndexInfo("s1", "f1")
    node2 = IndexInfo("s1", "f2")
    nodes = [node1, node2]
    establish_group_linkings(nodes)
    
    assert node1.g_prev is None
    assert node1.g_next is node2
    assert node2.g_prev is node1
    assert node2.g_next is None
def test_equality_with_linked_nodes():
    node1 = IndexInfo("s1", "f1")
    node2 = IndexInfo("s1", "f2", g_prev=node1)
    node3 = IndexInfo("s1", "f2", g_prev=node1)
    node4 = IndexInfo("s1", "f2")
    
    assert node2 == node3
    assert node2 != node4
def test_repr_method():
    node = IndexInfo("s1", "f1")
    node.g_next = IndexInfo("s1", "f2")
    node.g_prev = IndexInfo("s0", "f99")
    expected_repr = "s1/f1 (g_prev: s0/f99, g_next: s1/f2)"
    assert repr(node) == expected_repr
# Edge case: Empty list for establish_linkings
def test_establish_linkings_empty_list():
    assert establish_linkings([]) == []
    assert establish_group_linkings([]) == []
