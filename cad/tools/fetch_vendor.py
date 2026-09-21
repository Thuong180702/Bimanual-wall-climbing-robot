"""Download the three vendor actuator STEP files and verify their checksums.

The files are the manufacturers', not ours, so they are not in the repository.
Run this once before building the CAD.
"""
import hashlib, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "vendor")
API = "https://api.step.parts/v1/parts/%s"

PARTS = {
    "cubemars_ak80_64_kv80":
        "826b2603459d79cbcb64b54b02f7f137cb3656d240feb35aa6642fc9c3ce34f0",
    "cubemars_ak60_39_v3_kv80":
        "00e18639c4794eff5a83ea6930ae681433054ba1ed58fbb4b54434d303c4fe19",
    "cubemars_ak45_36_kv80":
        "d6d093884d7daab8517e38d75f5b86762dd1363e8c7a758280e77abbfd1e4686",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    bad = 0
    for pid, want in PARTS.items():
        dst = os.path.join(OUT, pid + ".step")
        if os.path.exists(dst) and \
                hashlib.sha256(open(dst, "rb").read()).hexdigest() == want:
            print("have  %s" % pid)
            continue
        rec = json.load(urllib.request.urlopen(API % pid))
        data = urllib.request.urlopen(rec["stepUrl"]).read()
        got = hashlib.sha256(data).hexdigest()
        if got != want:
            print("FAIL  %s: sha256 %s, expected %s" % (pid, got[:16], want[:16]))
            bad += 1
            continue
        open(dst, "wb").write(data)
        print("wrote %s  (%d bytes, sha256 ok)" % (dst, len(data)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
