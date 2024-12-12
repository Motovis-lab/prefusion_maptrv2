from prefusion.dataset.index_info import IndexInfo

def test_index_info_basic():
    ii = IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo.from_str('Scn/128'))
    assert ii.as_dict() == {'scene_id': 'Scn', 'frame_id': '127', 'prev': {'scene_id': 'Scn', 'frame_id': '126'}, 'next': {'scene_id': 'Scn', 'frame_id': '128'}}
    assert ii.prev.as_dict() == {'scene_id': 'Scn', 'frame_id': '126', 'prev': None, 'next': {'scene_id': 'Scn', 'frame_id': '127'}}
    assert ii.next.as_dict() == {'scene_id': 'Scn', 'frame_id': '128', 'prev': {'scene_id': 'Scn', 'frame_id': '127'}, 'next': None}

def test_index_info_modify():
    ii = IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo.from_str('Scn/128'))
    assert ii.as_dict() == {'scene_id': 'Scn', 'frame_id': '127', 'prev': {'scene_id': 'Scn', 'frame_id': '126'}, 'next': {'scene_id': 'Scn', 'frame_id': '128'}}
    ii.frame_id = '888'
    assert ii.prev.as_dict() == {'scene_id': 'Scn', 'frame_id': '126', 'prev': None, 'next': {'scene_id': 'Scn', 'frame_id': '888'}}
    assert ii.next.as_dict() == {'scene_id': 'Scn', 'frame_id': '128', 'prev': {'scene_id': 'Scn', 'frame_id': '888'}, 'next': None}


def test_index_info_eq():
    assert IndexInfo('Scn', '127') == IndexInfo('Scn', '127')
    assert IndexInfo('Scn', '127') != IndexInfo('Scn', '333')
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')) == IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'))
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')) == IndexInfo('Scn', '126', next=IndexInfo('Scn', '127')).next
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')) != IndexInfo('Scn', '126', next=IndexInfo('Scn', '127'))
    assert IndexInfo('Scn', '127', next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '127', next=IndexInfo('Scn', '128'))
    assert IndexInfo('Scn', '127', next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127')).prev
    assert IndexInfo('Scn', '127', next=IndexInfo('Scn', '128')) != IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127'))
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128'))
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128')) == IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'))).prev
    assert IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126'), next=IndexInfo('Scn', '128')) != IndexInfo('Scn', '128', prev=IndexInfo('Scn', '127', prev=IndexInfo('Scn', '126')))
