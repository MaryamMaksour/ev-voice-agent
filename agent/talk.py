"""Talk to the agent by voice from a terminal, through the self-hosted LiveKit server.

    python -m agent.talk --lead L003                 # microphone + speakers (headphones work best)
    python -m agent.talk --list-devices              # pick devices: --input-device 2 --output-device 4
    python -m agent.talk --lead L003 --audio-file caller.wav --save-audio agent.wav --seconds 60
                                                     # no sound card: send a WAV, save what the agent says

This is the same path a 3CX phone call takes later: a participant joins a LiveKit room, the agent
worker is dispatched to the room, and everything (audio, the agent's work, the logs) runs on the
server. Compare `python -m agent.main console`, which runs the agent inside the terminal with no
LiveKit server at all.

Needs in .env: LIVEKIT_URL (ws://<server-ip>:7880), LIVEKIT_API_KEY, LIVEKIT_API_SECRET.
Live captions of both sides are printed; Ctrl+C hangs up. The call record, logs and recording are
saved on the server in data/calls/ when the call ends.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
import wave
from datetime import datetime, timezone

from dotenv import load_dotenv

RATE = 48000
FRAME = RATE // 100  # 10 ms: the frame size LiveKit's echo canceller works on


def room_name(lead_id: str) -> str:
    return f"lead-{lead_id}-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"


def make_token(room: str, identity: str, name: str = "") -> str:
    from livekit import api

    return (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
            .with_identity(identity).with_name(name or identity)
            .with_grants(api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True))
            .to_jwt())


async def create_room(url: str, room: str, lead_id: str) -> None:
    """Create the room with the lead in its metadata; the agent worker is dispatched to it."""
    from livekit import api

    http = url.replace("wss://", "https://").replace("ws://", "http://")
    async with api.LiveKitAPI(url=http) as lk:
        await lk.room.create_room(api.CreateRoomRequest(
            name=room, empty_timeout=60, departure_timeout=5,
            metadata=json.dumps({"lead_id": lead_id, "source": "terminal"})))


class Speaker:
    """Agent audio waiting to be played (filled by LiveKit, drained by the sound card thread)."""

    def __init__(self, save_path: str | None = None) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._wav = None
        if save_path:
            self._wav = wave.open(save_path, "wb")
            self._wav.setnchannels(1), self._wav.setsampwidth(2), self._wav.setframerate(RATE)

    def push(self, pcm: bytes) -> None:
        with self._lock:
            self._buf.extend(pcm)
            if len(self._buf) > RATE * 2 * 10:  # never more than 10 s behind
                del self._buf[: len(self._buf) - RATE * 2 * 10]
        if self._wav:
            self._wav.writeframes(pcm)

    def pull(self, nbytes: int) -> bytes:
        with self._lock:
            chunk = bytes(self._buf[:nbytes])
            del self._buf[:nbytes]
        return chunk + b"\0" * (nbytes - len(chunk))

    def close(self) -> None:
        if self._wav:
            self._wav.close()


def list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


async def run(args: argparse.Namespace) -> int:
    from livekit import rtc

    url = args.url or os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    room_id = room_name(args.lead)
    await create_room(url, room_id, args.lead)
    token = make_token(room_id, args.identity, args.name)

    loop = asyncio.get_running_loop()
    room = rtc.Room(loop=loop)
    done = asyncio.Event()
    speaker = Speaker(args.save_audio)
    apm = None if args.audio_file or args.no_echo_cancel else rtc.AudioProcessingModule(
        echo_cancellation=True, noise_suppression=True, high_pass_filter=True, auto_gain_control=True)
    mic_track_sid: list[str] = []
    shown: set[str] = set()  # caption segments already printed
    players: list[asyncio.Task] = []
    started = time.monotonic()

    def say(who: str, text: str) -> None:
        print(f"[{time.monotonic() - started:6.1f}s] {who:>5}: {text}", flush=True)

    async def play_track(track: rtc.Track) -> None:
        audio = rtc.AudioStream(track, sample_rate=RATE, num_channels=1)
        try:
            async for ev in audio:
                speaker.push(bytes(ev.frame.data))
        finally:
            await audio.aclose()

    @room.on("track_subscribed")
    def _on_track(track: rtc.Track, pub, participant: rtc.RemoteParticipant) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            say("info", f"hearing {participant.identity}")
            players.append(asyncio.ensure_future(play_track(track)))

    @room.on("participant_connected")
    def _on_join(p: rtc.RemoteParticipant) -> None:
        say("info", f"{p.identity} joined")

    @room.on("participant_disconnected")
    def _on_leave(p: rtc.RemoteParticipant) -> None:
        say("info", f"{p.identity} left: call over")
        done.set()

    @room.on("disconnected")
    def _on_disconnected(*_a) -> None:
        done.set()

    async def caption(reader, participant_identity: str) -> None:
        attrs = reader.info.attributes or {}
        text = (await reader.read_all()).strip()
        key = attrs.get("lk.segment_id") or f"{participant_identity}:{text}"
        if not text or key in shown:  # the same segment can arrive more than once
            return
        shown.add(key)
        mine = attrs.get("lk.transcribed_track_id") in mic_track_sid or participant_identity == args.identity
        say("you" if mine else "agent", text)

    room.register_text_stream_handler("lk.transcription", lambda r, pid: asyncio.ensure_future(caption(r, pid)))

    say("info", f"joining {url} room {room_id} as lead {args.lead}")
    await room.connect(url, token)
    source = rtc.AudioSource(RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("caller-mic", source)
    pub = await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    mic_track_sid.append(pub.sid)
    say("info", "connected; waiting for the agent... (Ctrl+C to hang up)")

    frames: asyncio.Queue = asyncio.Queue(maxsize=200)
    stream = None
    sender: asyncio.Task | None = None

    if args.audio_file:
        sender = asyncio.ensure_future(send_wav(args.audio_file, source, rtc, args.wait_for_agent, room))
        if not args.no_playback and not args.save_audio:
            stream = open_output(speaker, args.output_device)
    else:
        stream = open_duplex(loop, frames, speaker, apm, rtc, args.input_device, args.output_device)

        async def pump() -> None:
            while True:
                await source.capture_frame(await frames.get())

        sender = asyncio.ensure_future(pump())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, done.set)
        except NotImplementedError:  # Windows: Ctrl+C raises KeyboardInterrupt instead
            pass
    try:
        if args.seconds:
            await asyncio.wait_for(done.wait(), timeout=args.seconds)
        else:
            await done.wait()
    except asyncio.TimeoutError:
        say("info", f"{args.seconds}s reached, hanging up")
    finally:
        for task in [sender, *players]:
            if task:
                task.cancel()
        await asyncio.gather(*(t for t in [sender, *players] if t), return_exceptions=True)
        if stream is not None:
            stream.stop(), stream.close()
        speaker.close()
        await room.disconnect()
        await source.aclose()
    say("info", f"call ended. Record, logs and recording: data/calls/{room_id}-*.json on the server")
    return 0


def open_duplex(loop, frames: asyncio.Queue, speaker: Speaker, apm, rtc, input_device, output_device):
    """Microphone in, agent out, in one sound card callback (10 ms blocks) so echo cancellation lines up."""
    import sounddevice as sd

    def callback(indata, outdata, nframes, _time, status) -> None:
        out = speaker.pull(nframes * 2)
        outdata[:] = out
        mic = bytes(indata)
        if apm is not None and nframes == FRAME:
            apm.process_reverse_stream(rtc.AudioFrame(out, RATE, 1, FRAME))
            frame = rtc.AudioFrame(mic, RATE, 1, FRAME)
            apm.process_stream(frame)
        else:
            frame = rtc.AudioFrame(mic, RATE, 1, nframes)
        try:
            loop.call_soon_threadsafe(frames.put_nowait, frame)
        except asyncio.QueueFull:
            pass

    stream = sd.RawStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="int16", latency="low",
                          device=(input_device, output_device), callback=callback)
    stream.start()
    if apm is not None:
        apm.set_stream_delay_ms(int((stream.latency[0] + stream.latency[1]) * 1000))
    return stream


def open_output(speaker: Speaker, output_device):
    import sounddevice as sd

    def callback(outdata, nframes, _time, status) -> None:
        outdata[:] = speaker.pull(nframes * 2)

    stream = sd.RawOutputStream(samplerate=RATE, blocksize=FRAME, channels=1, dtype="int16",
                                device=output_device, callback=callback)
    stream.start()
    return stream


async def send_wav(path: str, source, rtc, wait_for_agent: float, room) -> None:
    """Play a WAV file into the call as the caller's voice, in real time, after the agent's greeting."""
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: needs 16-bit PCM")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = w.readframes(w.getnframes())
    if channels != 1 or rate != RATE:
        resampler = rtc.AudioResampler(rate, RATE, num_channels=channels)
        frames = resampler.push(rtc.AudioFrame(pcm, rate, channels, len(pcm) // (2 * channels)))
        frames += resampler.flush()
        mono = bytearray()
        for f in frames:
            data = bytes(f.data)
            if channels == 1:
                mono += data
            else:  # keep the first channel
                step = 2 * channels
                mono += b"".join(data[i:i + 2] for i in range(0, len(data), step))
        pcm = bytes(mono)
    # wait for the agent to join and greet
    t0 = time.monotonic()
    while not room.remote_participants and time.monotonic() - t0 < 30:
        await asyncio.sleep(0.2)
    await asyncio.sleep(wait_for_agent)
    step = FRAME * 2
    silence = b"\0" * step
    for i in range(0, len(pcm), step):
        chunk = pcm[i:i + step]
        await source.capture_frame(rtc.AudioFrame(chunk + silence[len(chunk):], RATE, 1, FRAME))
    while True:  # keep sending silence so VAD sees the caller stop talking
        await source.capture_frame(rtc.AudioFrame(silence, RATE, 1, FRAME))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lead", default=os.getenv("DEFAULT_LEAD_ID", "L001"), help="CRM lead the agent calls")
    p.add_argument("--url", help="LiveKit server, default LIVEKIT_URL from .env")
    p.add_argument("--identity", default="terminal-caller")
    p.add_argument("--name", default="", help="display name in the room")
    p.add_argument("--list-devices", action="store_true", help="list sound devices and exit")
    p.add_argument("--input-device", type=lambda v: int(v) if v.isdigit() else v, default=None)
    p.add_argument("--output-device", type=lambda v: int(v) if v.isdigit() else v, default=None)
    p.add_argument("--no-echo-cancel", action="store_true", help="turn off echo cancellation (headphones)")
    p.add_argument("--audio-file", help="send this WAV as the caller instead of the microphone")
    p.add_argument("--wait-for-agent", type=float, default=6.0,
                   help="with --audio-file: seconds to let the agent greet before the WAV starts")
    p.add_argument("--save-audio", help="save what the agent says to this WAV")
    p.add_argument("--no-playback", action="store_true", help="with --audio-file: don't play the agent")
    p.add_argument("--seconds", type=float, default=0, help="hang up after this many seconds (0 = never)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    if args.list_devices:
        list_devices()
        return
    for var in ("LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.getenv(var):
            sys.exit(f"{var} is not set: copy .env.example to .env and fill in the LiveKit keys")
    try:
        sys.exit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
