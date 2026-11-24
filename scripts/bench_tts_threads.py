"""Controlled test: Kokoro TTS latency vs onnxruntime thread count, nothing else varied.

Run inside a job allocation, e.g.
    srun --jobid=<app job> --overlap --ntasks=1 .venv-app/bin/python -I scripts/bench_tts_threads.py
Measured 2026-10-22 on TC2N02 (8-core allocation, 104-core machine): default 9.61 s,
4 threads 0.81 s, 7 threads 0.92 s, 16 threads 3.15 s, 52 threads 8.91 s.
"""
import os, statistics, time, warnings
warnings.filterwarnings("ignore")
import onnxruntime as ort, psutil
from kokoro_onnx import Kokoro

MODEL, VOICES = "models/tts/kokoro-v1.0.onnx", "models/tts/voices-v1.0.bin"
TEXT = "The person is changing a tire on a car."
print(f"os.cpu_count() (machine)         : {os.cpu_count()}")
print(f"sched_getaffinity (job may use)  : {len(os.sched_getaffinity(0))}")
print(f"SLURM_CPUS_PER_TASK              : {os.environ.get('SLURM_CPUS_PER_TASK')}")
ort.set_default_logger_severity(4)  # silence the affinity error spam for timing

def run(threads):
    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
    before = psutil.Process().num_threads()
    sess = ort.InferenceSession(MODEL, sess_options=opts, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, VOICES)
    created = psutil.Process().num_threads() - before
    for _ in range(2):
        k.create(TEXT, voice="af_heart", lang="en-us")          # warm-up, not timed
    ts = []
    for _ in range(5):
        t = time.perf_counter(); k.create(TEXT, voice="af_heart", lang="en-us"); ts.append(time.perf_counter() - t)
    del k, sess
    return created, statistics.median(ts), min(ts), max(ts)

print(f"\n{'intra_op threads':<18}{'threads created':>16}{'median s':>10}{'min':>7}{'max':>7}")
for th in [None, 1, 2, 4, 7, 8, 16, 52, 104]:
    created, med, lo, hi = run(th)
    label = "default" if th is None else str(th)
    print(f"{label:<18}{created:>16}{med:>10.3f}{lo:>7.3f}{hi:>7.3f}", flush=True)
