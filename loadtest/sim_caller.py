"""A simulated caller: joins a LiveKit room, waits for the agent's greeting, then plays
pre-recorded answers and measures, for every turn, the time from the end of its own
speech to the first audible audio from the agent (what a real caller experiences)."""

from __future__ import annotations

import asyncio
import logging
import time
import wave
from array import array
from dataclasses import dataclass, field
from pathlib import Path

from livekit import api, rtc

logger = logging.getLogger("loadtest.caller")

FRAME_MS = 10
SPEECH_RMS = 300          # int16 RMS above this = agent is speaking
SILENCE_AFTER_SPEECH = 1.0  # seconds of quiet = agent finished its reply


def load_wav(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, f"{path}: need 16-bit mono"
        return w.getframerate(), w.readframes(w.getnframes())


def rms(pcm: bytes) -> float:
    samples = array("h", pcm)
    if not samples:
        return 0.0
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


@dataclass
class CallResult:
    room: str
    latencies: list[float] = field(default_factory=list)
    timeouts: int = 0
    error: str | None = None


class SimCaller:
    def __init__(self, url: str, api_key: str, api_secret: str, room: str, utterances: list[Path],
                 reply_timeout: float = 20.0) -> None:
        self.url, self.room_name = url, room
        self.token = (
            api.AccessToken(api_key, api_secret)
            .with_identity(f"sim-{room}")
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .to_jwt()
        )
        self.audio = [load_wav(p) for p in utterances]
        self.sample_rate = self.audio[0][0]
        self.reply_timeout = reply_timeout
        self._agent_speaking = asyncio.Event()
        self._agent_quiet = asyncio.Event()
        self._first_audio_at = 0.0
        self._play_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._played = asyncio.Event()

    async def _watch_agent(self, track: rtc.Track) -> None:
        stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1, frame_size_ms=FRAME_MS)
        last_loud = 0.0
        try:
            await self._read_agent(stream, last_loud)
        finally:
            await stream.aclose()

    async def _read_agent(self, stream: rtc.AudioStream, last_loud: float) -> None:
        async for ev in stream:
            now = time.perf_counter()
            if rms(bytes(ev.frame.data)) > SPEECH_RMS:
                last_loud = now
                if not self._agent_speaking.is_set():
                    self._first_audio_at = now
                    self._agent_quiet.clear()
                    self._agent_speaking.set()
            elif self._agent_speaking.is_set() and now - last_loud > SILENCE_AFTER_SPEECH:
                self._agent_speaking.clear()
                self._agent_quiet.set()

    async def _mic(self, source: rtc.AudioSource) -> None:
        """Continuous mic: speech when queued, silence otherwise (the VAD needs silence)."""
        samples = self.sample_rate * FRAME_MS // 1000
        silence = b"\x00\x00" * samples
        while True:
            pcm = self._play_queue.get_nowait() if not self._play_queue.empty() else None
            chunks = [pcm[i : i + samples * 2] for i in range(0, len(pcm), samples * 2)] if pcm else [silence]
            for chunk in chunks:
                chunk = chunk.ljust(samples * 2, b"\x00")
                await source.capture_frame(rtc.AudioFrame(chunk, self.sample_rate, 1, samples))
            if pcm:
                self._played.set()

    async def _say(self, pcm: bytes) -> float:
        self._played.clear()
        await self._play_queue.put(pcm)
        await self._played.wait()
        return time.perf_counter()

    async def run(self, start_delay: float = 0.0) -> CallResult:
        result = CallResult(room=self.room_name)
        await asyncio.sleep(start_delay)
        room = rtc.Room()
        source: rtc.AudioSource | None = None
        tasks: list[asyncio.Task] = []

        @room.on("track_subscribed")
        def _on_track(track: rtc.Track, *_):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                tasks.append(asyncio.create_task(self._watch_agent(track)))

        try:
            await room.connect(self.url, self.token)
            source = rtc.AudioSource(self.sample_rate, 1, queue_size_ms=50)
            track = rtc.LocalAudioTrack.create_audio_track("mic", source)
            await room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            )
            tasks.append(asyncio.create_task(self._mic(source)))

            # greeting: the agent speaks first
            await asyncio.wait_for(self._agent_speaking.wait(), timeout=60)
            await asyncio.wait_for(self._agent_quiet.wait(), timeout=60)

            for _, pcm in self.audio:
                ended_at = await self._say(pcm)
                try:
                    await asyncio.wait_for(self._agent_speaking.wait(), timeout=self.reply_timeout)
                    result.latencies.append(self._first_audio_at - ended_at)
                    await asyncio.wait_for(self._agent_quiet.wait(), timeout=60)
                except asyncio.TimeoutError:
                    result.timeouts += 1
                await asyncio.sleep(0.3)
        except Exception as e:  # report, don't crash the whole load run
            result.error = repr(e)
            logger.warning("%s failed: %r", self.room_name, e)
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if source is not None:
                await source.aclose()
            await room.disconnect()
        return result
