"""Command line: send / receive live, encode / decode WAV, simulate."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np

from .params import FS, Params
from .protocol import Receiver, Sender


def _params(a) -> Params:
    return Params(step=a.step)


def _pass(a):
    return a.passphrase or os.environ.get("TX_AUDIO_PASSPHRASE") or None


def _digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _log(verbose):
    return (lambda m: print(m, flush=True)) if verbose else None


def _write(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def cmd_encode(a):
    data = open(a.source, "rb").read()
    p = _params(a)
    s = Sender(data, p, passphrase=_pass(a), use_feedback=False, rounds=a.rounds, log=_log(a.verbose))
    chunks = []
    while not s.finished:
        chunks.append(s.next_chunk(4800))
    x = np.concatenate(chunks)
    from .dsp import write_wav
    write_wav(a.out, x, p.fs)
    print(f"wrote {a.out}: {len(x) / p.fs:.1f} s of music for {len(data)} bytes "
          f"({s.total} blocks x {a.rounds} rounds), sha256 {_digest(data)}")


def cmd_decode(a):
    from .dsp import read_wav
    p = _params(a)
    x = read_wav(a.wav, p.fs)
    r = Receiver(p, passphrase=_pass(a), send_feedback=False, log=_log(a.verbose))
    for i in range(0, len(x), 4800):
        r.feed(x[i: i + 4800])
    r.flush()
    out = r.result()
    if out is None:
        miss = r.missing()
        print(f"incomplete: {len(r.blocks)}/{r.total or '?'} blocks; missing {miss[:20]}",
              file=sys.stderr)
        return 1
    _write(a.target, out)
    print(f"wrote {a.target}: {len(out)} bytes, sha256 {_digest(out)}")
    return 0


def cmd_send(a):
    from .live import run_live
    data = open(a.source, "rb").read()
    p = _params(a)
    s = Sender(data, p, passphrase=_pass(a), log=_log(a.verbose))
    print(f"sending {len(data)} bytes in {s.total} blocks (~{p.net_bps():.0f} bit/s), "
          f"sha256 {_digest(data)}")
    shown = [False]

    def status():
        if s.safety_code and not shown[0]:
            print(f"key agreed, safety code {s.safety_code} (check the receiver shows the same)")
            shown[0] = True
        if not a.verbose:
            print(f"  {len(s.acked)}/{s.total} blocks acknowledged", flush=True)

    run_live(s, p.fs, lambda: s.finished, device=a.device, timeout_s=a.timeout, status=status)
    if s.done:
        print("receiver confirmed the whole file")
        return 0
    print("finished without confirmation" + (" (no key from receiver)" if s.failed else ""))
    return 1


def cmd_receive(a):
    from .live import run_live
    p = _params(a)
    r = Receiver(p, passphrase=_pass(a), log=_log(a.verbose))
    state = {"written": False, "t_done": None, "code": False}

    def status():
        if r.safety_code and not state["code"]:
            print(f"key agreed, safety code {r.safety_code}")
            state["code"] = True
        if not a.verbose and r.total:
            print(f"  {len(r.blocks)}/{r.total} blocks", flush=True)

    def stop():
        if r.complete and not state["written"]:
            out = r.result()
            _write(a.target, out)
            print(f"wrote {a.target}: {len(out)} bytes, sha256 {_digest(out)}")
            state["written"] = True
            state["t_done"] = time.time()
        # Linger so a sender that missed DONE can prompt us again.
        return state["t_done"] is not None and time.time() - state["t_done"] > a.linger

    print("listening... (Ctrl-C to stop)")
    try:
        run_live(r, p.fs, stop, device=a.device, timeout_s=a.timeout, status=status)
    except KeyboardInterrupt:
        pass
    return 0 if state["written"] else 1


def cmd_simulate(a):
    from .channel import ChannelConfig, simulate_duplex
    from .dsp import write_wav
    data = open(a.source, "rb").read()
    p = _params(a)
    echoes = [(9, 0.5), (23, 0.35), (41, 0.2)] if a.reverb > 0 else []
    fwd = ChannelConfig(gain=a.gain, delay_ms=a.distance / 343 * 1000, echoes=echoes,
                        reverb_rt60=a.reverb, reverb_level=0.4 if a.reverb else 0.0,
                        drift_ppm=a.drift_ppm, wobble_db=a.wobble_db, seed=1)
    t0 = time.time()
    res = simulate_duplex(data, p, forward=fwd, noise_db=a.noise_db, record=bool(a.record_dir),
                          seed=None, passphrase=_pass(a))
    if a.verbose:
        print("\n".join(sorted(res.tx_log + res.rx_log, key=lambda m: 0)))
    if a.record_dir:
        os.makedirs(a.record_dir, exist_ok=True)
        write_wav(os.path.join(a.record_dir, "receiver_mic.wav"), res.rx_mic, p.fs)
        write_wav(os.path.join(a.record_dir, "sender_mic.wav"), res.tx_mic, p.fs)
    print(f"{'OK' if res.ok else 'FAILED'}: {len(data)} bytes in {res.seconds:.1f} s of audio "
          f"({len(data) * 8 / res.seconds:.1f} bit/s overall), handshake {res.handshake_s or 0:.1f} s, "
          f"{res.frames_sent} frames for {res.blocks} blocks, safety codes {res.safety_codes}, "
          f"sim took {time.time() - t0:.1f} s")
    if res.ok:
        _write(a.target, data)
    return 0 if res.ok else 1


def cmd_devices(a):
    try:
        import sounddevice as sd
    except OSError as e:
        raise SystemExit(f"PortAudio not available: {e}")
    print(sd.query_devices())
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tx-audio", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--step", type=float, default=0.1,
                        help="seconds per 4-note arpeggio (one byte); both sides must match")
        sp.add_argument("--passphrase", help="optional shared secret (or TX_AUDIO_PASSPHRASE)")
        sp.add_argument("-v", "--verbose", action="store_true")

    sp = sub.add_parser("send", help="play a file as music and wait for confirmation")
    sp.add_argument("source")
    sp.add_argument("--device", default=None)
    sp.add_argument("--timeout", type=float, default=None)
    common(sp)
    sp.set_defaults(fn=cmd_send)

    sp = sub.add_parser("receive", help="listen, answer with tones, write the file")
    sp.add_argument("target")
    sp.add_argument("--device", default=None)
    sp.add_argument("--timeout", type=float, default=None)
    sp.add_argument("--linger", type=float, default=15.0,
                    help="seconds to keep answering after the file is complete")
    common(sp)
    sp.set_defaults(fn=cmd_receive)

    sp = sub.add_parser("encode", help="write the music to a WAV (no feedback)")
    sp.add_argument("source")
    sp.add_argument("out")
    sp.add_argument("--rounds", type=int, default=2, help="send every block this many times")
    common(sp)
    sp.set_defaults(fn=cmd_encode)

    sp = sub.add_parser("decode", help="recover a file from a WAV recording")
    sp.add_argument("wav")
    sp.add_argument("target")
    common(sp)
    sp.set_defaults(fn=cmd_decode)

    sp = sub.add_parser("simulate", help="full duplex send/receive through a simulated room")
    sp.add_argument("source")
    sp.add_argument("target")
    sp.add_argument("--noise-db", type=float, default=-50.0, help="background noise, dBFS RMS")
    sp.add_argument("--reverb", type=float, default=0.4, help="RT60 seconds (0 = none)")
    sp.add_argument("--drift-ppm", type=float, default=100.0)
    sp.add_argument("--wobble-db", type=float, default=3.0)
    sp.add_argument("--gain", type=float, default=0.3, help="speaker-to-mic gain")
    sp.add_argument("--distance", type=float, default=2.0, help="metres between devices")
    sp.add_argument("--record-dir", help="save both microphones' audio as WAV here")
    common(sp)
    sp.set_defaults(fn=cmd_simulate)

    sp = sub.add_parser("devices", help="list sound devices")
    sp.set_defaults(fn=cmd_devices)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
