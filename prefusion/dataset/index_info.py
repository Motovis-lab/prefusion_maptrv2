from typing import Optional, List


__all__ = ["IndexInfo"]


class IndexInfo:
    def __init__(
        self,
        scene_id: str,
        frame_id: str,
        prev: Optional["IndexInfo"] = None,
        next: Optional["IndexInfo"] = None,
        g_prev: Optional["IndexInfo"] = None,
        g_next: Optional["IndexInfo"] = None,
    ):
        """Object that stores scene_id and frame_id as well as the adjacent linking relationships.

        Parameters
        ----------
        scene_id : str
            scene id
        frame_id : str
            frame id
        prev : IndexInfo, optional
            previous index_info, by default None
        next : IndexInfo, optional
            next index_info, by default None
        g_prev : IndexInfo, optional
            previous index_info within the same group, by default None
        g_next : IndexInfo, optional
            next index_info within the same group, by default None
        """
        self.scene_id = scene_id
        self.frame_id = frame_id
        self.prev = prev
        self.next = next
        self.g_prev = g_prev
        self.g_next = g_next
        if prev:
            prev.next = self
        if next:
            next.prev = self
        if g_prev:
            g_prev.g_next = self
        if g_next:
            g_next.g_prev = self

    def __repr__(self) -> str:
        return f"{self.scene_id}/{self.frame_id} (g_prev: {self.g_prev.scene_frame_id if self.g_prev else None}, g_next: {self.g_next.scene_frame_id if self.g_next else None})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IndexInfo):
            return False
        if not self.two_index_info_are_equal(self.prev, other.prev):
            return False
        if not self.two_index_info_are_equal(self.next, other.next):
            return False
        if not self.two_index_info_are_equal(self.g_prev, other.g_prev):
            return False
        if not self.two_index_info_are_equal(self.g_next, other.g_next):
            return False
        if self.scene_frame_id != other.scene_frame_id:
            return False

        return True

    @staticmethod
    def two_index_info_are_equal(a_index_info, another_index_info):
        if (a_index_info is None) ^ (another_index_info is None):
            return False
        if a_index_info is not None and another_index_info is not None and a_index_info.scene_frame_id != another_index_info.scene_frame_id:
            return False
        return True

    @property
    def scene_frame_id(self) -> str:
        return f"{self.scene_id}/{self.frame_id}"

    def as_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "prev": {"scene_id": self.prev.scene_id, "frame_id": self.prev.frame_id} if self.prev else None,
            "next": {"scene_id": self.next.scene_id, "frame_id": self.next.frame_id} if self.next else None,
            "g_prev": {"scene_id": self.g_prev.scene_id, "frame_id": self.g_prev.frame_id} if self.g_prev else None,
            "g_next": {"scene_id": self.g_next.scene_id, "frame_id": self.g_next.frame_id} if self.g_next else None,
        }

    @classmethod
    def from_str(
        cls,
        index_str: str,
        prev: Optional["IndexInfo"] = None,
        next: Optional["IndexInfo"] = None,
        g_prev: Optional["IndexInfo"] = None,
        g_next: Optional["IndexInfo"] = None,
        sep: str = "/",
    ):
        """Create IndexInfo instance from index string.

        Parameters
        ----------
        index_str : str
            expected format: `scene_id/frame_id`
        prev : IndexInfo, optional
            previous index_info, by default None
        next : IndexInfo, optional
            next index_info, by default None
        g_prev : IndexInfo, optional
            previous index_info within the same group, by default None
        g_next : IndexInfo, optional
            next index_info within the same group, by default None
        sep : str, optional
            separator, by default "/"
        """
        scene_id, frame_id = index_str.split(sep)
        return IndexInfo(scene_id, frame_id, prev=prev, next=next, g_prev=g_prev, g_next=g_next)


def establish_linkings(index_info_list: List["IndexInfo"], group_linking=False) -> List["IndexInfo"]:
    attr_prev = "g_prev" if group_linking else "prev"
    attr_next = "g_next" if group_linking else "next"
    prev = None
    for i in range(len(index_info_list)):
        cur = index_info_list[i]
        setattr(cur, attr_prev, prev)
        if prev is not None:
            setattr(prev, attr_next, cur)
        prev = cur
    return index_info_list


def establish_group_linkings(index_info_list: List["IndexInfo"]) -> List["IndexInfo"]:
    return establish_linkings(index_info_list, group_linking=True)
