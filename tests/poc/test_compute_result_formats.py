import base64

from vllm.poc.server.compute import _extract_vectors_b64


def test_extract_vectors_b64_passthrough():
    res = {"vectors_b64": ["AAA=", "BBB="]}
    assert _extract_vectors_b64(res) == ["AAA=", "BBB="]


def test_extract_vectors_b64_from_binary():
    raw0 = b"\x01\x02"
    raw1 = bytearray(b"\x03\x04")
    raw2 = memoryview(b"\x05\x06")

    res = {"vectors_bin": [raw0, raw1, raw2]}
    out = _extract_vectors_b64(res)

    assert out == [
        base64.b64encode(b"\x01\x02").decode("ascii"),
        base64.b64encode(b"\x03\x04").decode("ascii"),
        base64.b64encode(b"\x05\x06").decode("ascii"),
    ]
