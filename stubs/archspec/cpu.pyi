class Microarchitecture:
    name: str
    vendor: str
    generation: int
    features: set[str]

def host() -> Microarchitecture: ...
