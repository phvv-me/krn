"""The part of PyObjC's Foundation that the Apple profiler calls: the `NSRange` a counter buffer resolves a span with."""

class NSRange:
    location: int
    length: int

def NSMakeRange(location: int, length: int) -> NSRange: ...
