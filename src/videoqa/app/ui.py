"""Gradio front end: layout and event wiring only -- all logic lives in MultimodalAssistant."""

from __future__ import annotations

import gradio as gr

from videoqa.app.assistant import InputError, MultimodalAssistant, Reply, Source

CSS = """
#title {text-align:center} #title h1 {margin-bottom:0}
#status {font-size:0.95em; opacity:0.85}
"""


class AssistantUI:
    def __init__(self, assistant: MultimodalAssistant, hands_free: bool = True,
                 title: str = "InternVL3.5 · Video QA") -> None:
        self.assistant = assistant
        self.hands_free = hands_free and assistant.endpointer is not None
        self.title = title

    # -- formatting ----------------------------------------------------------
    def _render(self, reply: Reply, history: list) -> tuple:
        history = (history or []) + [
            {"role": "user", "content": reply.question},
            {"role": "assistant", "content": reply.answer},
        ]
        t = " · ".join(f"{k.upper()} {v * 1000:.0f} ms" for k, v in reply.timings.items())
        info = f"**{self.assistant.vqa.describe()}** · {reply.detail} · {t}"
        return history, reply.speech.as_gradio(), reply.frames, info, ""

    @staticmethod
    def _skip(n: int = 5) -> tuple:
        return tuple(gr.skip() for _ in range(n))

    # -- handlers ------------------------------------------------------------
    def on_frame(self, frame):
        return f"🎥 {self.assistant.push_camera_frame(frame)} frames buffered"

    def on_ask(self, question, source, image, video, history):
        try:
            return self._render(self.assistant.ask(question, Source(source), image, video), history)
        except InputError as e:
            return history, None, [], f"⚠️ {e}", question

    def on_audio_chunk(self, chunk, source, image, video, history):
        """Hands-free: called every ~0.25 s with fresh mic audio."""
        if chunk is None:
            return self._skip()
        utterances = self.assistant.hear(*chunk)
        if not utterances:
            return self._skip()
        try:
            reply = self.assistant.ask_spoken(utterances[-1], Source(source), image, video)
        except InputError as e:
            return history, None, [], f"⚠️ {e}", gr.skip()
        return self._render(reply, history)

    def on_recording(self, audio):
        """Push-to-talk fallback: transcribe the whole recording into the textbox."""
        if audio is None:
            return gr.skip(), "no audio"
        heard = self.assistant.transcribe(*audio)
        return heard.text, f"ASR {heard.seconds * 1000:.0f} ms"

    # -- layout --------------------------------------------------------------
    def build(self) -> gr.Blocks:
        with gr.Blocks(title=self.title, css=CSS, theme=gr.themes.Soft()) as app:
            gr.Markdown(f"# {self.title}\nVideos · images · text · speech — "
                        f"InternVL3.5 with Nemotron ASR and {self.assistant.tts.name}", elem_id="title")
            with gr.Row():
                with gr.Column(scale=1):
                    source = gr.Radio([s.value for s in Source], value=Source.CAMERA.value, label="Visual input")
                    cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Live camera")
                    buffered = gr.Markdown("🎥 0 frames buffered", elem_id="status")
                    image = gr.Image(sources=["upload", "clipboard"], type="numpy", label="Image")
                    video = gr.Video(sources=["upload"], label="Video file")
                with gr.Column(scale=1):
                    chat = gr.Chatbot(label="Conversation", height=340)
                    mic_label = ("🎙️ Click once to start listening — just speak; it answers when you pause"
                                 if self.hands_free else "🎙️ Hold to record, release to transcribe")
                    mic = gr.Audio(sources=["microphone"], type="numpy", streaming=self.hands_free,
                                   label=mic_label)
                    question = gr.Textbox(label="Or type a question", placeholder="What is the person doing?")
                    ask = gr.Button("Ask", variant="primary")
                    spoken = gr.Audio(label="Spoken answer", autoplay=True, interactive=False)
                    info = gr.Markdown(elem_id="status")
                    seen = gr.Gallery(label="Frames the model saw", columns=8, height=110)

            cam.stream(self.on_frame, cam, buffered, stream_every=0.25, concurrency_limit=1)
            ins = [question, source, image, video, chat]
            outs = [chat, spoken, seen, info, question]
            ask.click(self.on_ask, ins, outs)
            question.submit(self.on_ask, ins, outs)
            if self.hands_free:
                mic.stream(self.on_audio_chunk, [mic, source, image, video, chat], outs,
                           stream_every=0.25, concurrency_limit=1, trigger_mode="always_last")
            else:
                mic.stop_recording(self.on_recording, mic, [question, info]).then(self.on_ask, ins, outs)
        return app

    def launch(self, host: str, port: int, auth: tuple[str, str] | None) -> None:
        self.build().queue(default_concurrency_limit=1).launch(
            server_name=host, server_port=port, show_error=True, auth=auth)
