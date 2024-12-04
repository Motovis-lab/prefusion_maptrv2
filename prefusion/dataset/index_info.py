__all__ = ["IndexInfo"]

class IndexInfo:
    def __init__(self, scene_id: str, frame_id: str, prev: "IndexInfo" = None, next: "IndexInfo" = None):
        self.scene_id = scene_id
        self.frame_id = frame_id
        self.prev = prev
        self.next = next
        if prev:
            prev.next = self
        if next:
            next.prev = self

    def __repr__(self) -> str:
        return f"{self.scene_id}/{self.frame_id} (prev: {self.prev.scene_frame_id if self.prev else None}, next: {self.next.scene_frame_id if self.next else None})"

    def __eq__(self, other: "IndexInfo") -> bool:
        if other is None:
            return False
        if self.scene_frame_id != other.scene_frame_id:
            return False
        if self.prev is None and other.prev is not None:
            return False
        if self.next is None and other.next is not None:
            return False
        if self.prev is not None and other.prev is not None and self.prev.scene_frame_id != other.prev.scene_frame_id:
            return False
        if self.next is not None and other.next is not None and self.next.scene_frame_id != other.next.scene_frame_id:
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
        }

    @classmethod
    def from_str(self, index_str: str, prev: "IndexInfo" = None, next: "IndexInfo" = None, sep: str = "/"):
        scene_id, frame_id = index_str.split(sep)
        return IndexInfo(scene_id, frame_id, prev=prev, next=next)
