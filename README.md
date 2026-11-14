# tx-audio

Send a file across a room as music. The sender turns the file into a
continuous, ever-changing electronic track; the receiver listens, and
answers with short tone bursts to say which parts arrived and which need
resending. Each send negotiates its own encoding, so no two sends sound or
decode alike.

This is an exploration of what is possible over plain speakers and
microphones, not a fast link: expect roughly 45 bit/s of file data at the
default tempo (about 3.5 minutes per kilobyte, including the handshake).

## Install

```bash
python3 -m pip install -e .            # numpy, reedsolo, cryptography
python3 -m pip install -e '.[live]'    # + sounddevice for speakers/mics
```

Live mode needs PortAudio (`brew install portaudio` on macOS,
`apt install libportaudio2` on Debian/Ubuntu). File mode and the simulator
do not.

## Use

One source file in, one target file out.

```bash
# Live, on two machines (start the receiver first):
tx-audio receive received.bin
tx-audio send photo.jpg

# Both print a 6-digit safety code once they agree on a key; they should match.
# Add the same --passphrase on both sides to hide even the handshake.

# Without a return channel: render to WAV, play or record it, decode later.
tx-audio encode photo.jpg song.wav --rounds 2 --passphrase "correct horse"
tx-audio decode recording.wav received.bin --passphrase "correct horse"

# Full duplex through a simulated room (both mics hear both speakers):
tx-audio simulate photo.jpg received.bin --reverb 0.6 --drift-ppm 300 \
    --noise-db -35 --record-dir out/
```

`--step` sets the tempo (seconds per four-note arpeggio, i.e. per byte).
Both sides must use the same value. `tx-audio devices` lists sound cards;
pass `--device` to pick one.

## How it works

**Data as music.** Each byte becomes one step of a four-voice arpeggio.
Three octaves of the pentatonic scale (A4 to A7) are split into four
registers of four notes; each voice picks one of its four notes, so each
voice carries 2 bits and the four together carry a byte. The voices play a
quarter step apart as a fast arpeggio, and each note rings for a full step,
so the receiver can use a long, sharp analysis window per voice even though
notes change 40 times a second. Any mix of pentatonic notes sounds
consonant, which is what lets random-looking data still sound like music.
Underneath, a kick, hats, bass line and pad follow a chord progression; the
accompaniment carries no data and stays out of the data band (bass and pad
below 350 Hz, hats above 9 kHz). The progression, drum and bass patterns
are picked at random per send and change every four bars.

**Framing and error correction.** A frame is a 4-step sync motif, a
Reed-Solomon protected header (block number, block count, length) and a
Reed-Solomon protected payload of 64 file bytes plus a 16-byte
authentication tag. The demodulator also reports how confident it was in
each byte; when plain decoding fails, the least confident bytes are marked
as erasures, which doubles what Reed-Solomon can repair (16 parity bytes
fix 8 unknown errors or 16 flagged ones). Frame timing is found by scoring
the sync motif, and clock drift between the two devices is estimated from
the spacing of successive frames.

**Feedback.** The receiver never plays music. It answers in 16-FSK tones
between 4.4 and 6.2 kHz, a band the music leaves empty; the tone
frequencies sit in the gaps between the 2nd to 6th harmonics of the data
notes, so speaker distortion of the music does not land on them. Messages
are ACK(block), NAK(block), DONE and KEY, each Reed-Solomon coded (about
1 s for an ACK, 6 s for the one-off KEY). Tones are 60 ms with a 15 ms
guard, and the decoder subtracts the reverb tail of the previous two tones
before deciding, which is what made the KEY survive a very echoey room. The
sender keeps playing while it listens, so its own mic hears its music
loudly; the band split is what keeps the two apart.

**Protocol.**
1. Sender plays an intro bar, then a HELLO frame with an ephemeral X25519
   public key and a random salt, then short idle frames while it waits
   (HELLO repeats if nobody answers).
2. Receiver answers with its own X25519 key in tones. Both derive the
   session with HKDF and show the same safety code.
3. Sender plays every block once. Receiver ACKs good blocks and NAKs
   damaged ones.
4. Sender resends NAKed blocks, then anything never ACKed, then plays idle
   frames, which prompt the receiver to NAK whatever is still missing.
5. Receiver says DONE once it has the whole file; sender plays an outro.

## Every send is encoded differently

The session key (from the X25519 exchange, plus the passphrase if given)
decides:

- the 16 sync motifs, which rotate frame by frame,
- the order the four voices play in the arpeggio,
- a keystream XORed onto every coded byte, so the byte-to-note mapping
  changes on every step of every frame,
- the order and preamble of the feedback tones,
- the ChaCha20-Poly1305 key that encrypts and authenticates each block.

So the same file sent twice produces unrelated note sequences, and there is
no fixed pattern to fingerprint once the session starts.

What this does and does not protect against:

- **Passive eavesdropper, live mode:** cannot decode the file, even with a
  perfect recording of both sides. It never has either private key.
- **The handshake is recognisable** without a passphrase: the HELLO frame
  and the KEY reply use a fixed public scheme, so a listener who knows this
  program can tell a transfer is starting (but learns nothing about the
  file). With `--passphrase`, the handshake scheme is derived from the
  passphrase too, and nothing on air is fixed.
- **Active attacker (man in the middle):** X25519 alone does not stop
  someone who plays their own tones and music in the room. Compare the
  safety codes, or use a passphrase, which is mixed into the key so an
  attacker without it derives the wrong keys. A weak passphrase can be
  guessed offline from a recording (it is stretched with scrypt, which
  slows this down but does not prevent it).
- **File mode** has no return channel, so there is no key exchange: the
  key comes from the salt in the HELLO plus the passphrase. Without a
  passphrase this is obfuscation, not secrecy: anyone with this program
  can decode the WAV.
- **Traffic analysis:** the length of the transfer and the timing of
  feedback bursts still reveal roughly how big the file is and how many
  retries happened.

## Results

All numbers below are from the built-in room simulator (`txaudio/channel.py`),
not real speakers. The simulator models gain, early echoes plus a reverb
tail, sample-clock drift between the two sound cards, slow loudness wobble,
white noise, dropouts, and, in duplex runs, both microphones hearing both
speakers.

**Throughput, full duplex with handshake and acknowledgements** (default
tempo, 0.1 s per byte-step, 45 bit/s raw payload rate):

| File | Room | Time on air | Effective rate |
|---|---|---|---|
| 256 B | quiet room (RT60 0.4 s, 100 ppm drift, noise -50 dBFS) | 71 s | 29 bit/s |
| 256 B | harsh room (RT60 0.9 s wet, -800 ppm, 8 dB wobble, noise -30 dBFS) | 73 s | 28 bit/s |
| 1 KB | quiet room | 207 s | 40 bit/s |
| 1 KB | harsh room | 207 s | 40 bit/s |

The handshake (intro bar, HELLO, KEY reply) costs about 19 s per send, which
is why small files come in well under the raw rate. No block needed a resend
in these runs; the NAK path is exercised by a test that blanks 2.5 s mid-send.

**Robustness, one-way decode** (`python3 tools/sweep.py`, 6 blocks; "repaired"
is the share of payload bytes Reed-Solomon had to fix, out of a 16.7% budget):

| Condition | 0.1 s/step (45 bit/s) | 0.064 s/step (71 bit/s) |
|---|---|---|
| white noise at -20 dBFS | ok, 0% | ok, 0% |
| white noise at -15 dBFS (louder than the music at the mic) | ok, 2.8% | fails |
| reverb RT60 0.4 s | ok, 1.2% | ok, 1.2% |
| reverb RT60 0.8 s | ok, 1.0% | 5 of 6 blocks |
| reverb RT60 1.2 s, very wet | ok, 0.9% | fails |
| clock drift +1000 / -2000 ppm | ok, 0% | ok, 0% |
| loudness wobble 10 dB | ok, 0% | ok, 0% |
| a second tx-audio sender 6 dB quieter | ok, 0.4% | ok, 0% |
| everything at once (RT60 0.6, 500 ppm, 6 dB wobble, -30 dBFS) | ok, 1.0% | ok, 1.9% |

So the default tempo has margin everywhere tested; the faster tempo works
in a quiet, dry room but has little left for reverb or loud noise.

## Left to try on real speakers

- Laptop and phone speakers roll off the low end and distort loud highs;
  the feedback band (4.4 to 6.2 kHz) and top data notes (up to 3.5 kHz)
  need checking on cheap hardware.
- Automatic gain control and noise suppression on laptop mics (especially
  in conferencing modes) may treat the music as noise. Disable them if
  decoding struggles.
- Real full-duplex echo: the simulator's self-path is simple. A sender that
  hears its own music very loudly may need its mic pointed away or lower
  input gain.
- Bluetooth speakers add 100 to 300 ms of latency and their own codec; the
  protocol tolerates the latency, the codec is untested.
- Tuning: tempo, payload size and parity are parameters (`--step`, and
  `Params` in code); a real room may let 0.08 s per step work.

## Tests

```bash
python3 -m pytest -q -m "not slow"   # unit + end-to-end, about a minute
python3 -m pytest -q                 # also harsher rooms and a fake sound card
```
