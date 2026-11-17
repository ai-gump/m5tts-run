import subprocess, os, time, json, sys

BIN = "./audio.cpp/build/bin/audiocpp_cli"
TEXT = "人工智能正在改变我们生活和工作的方式，语音合成技术让机器可以用自然的声音与人交流。"
for run in range(1, 4):
    out = f"outputs/ac_ttfa{run}.wav"
    cmd = [BIN, "--family", "m5_tts", "--task", "clon", "--backend", "metal",
           "--model", "m5-voiceclone-f16.gguf",
           "--voice-ref", "M5-TTS/assets/aslp_default_voice.wav",
           "--reference-text", "猪携带该疾病，然后通过蚊子传染给人类。",
           "--language", "zh", "--text", TEXT, "--threads", "8",
           "--mode", "streaming", "--out", out]
    if os.path.exists(out): os.remove(out)
    t0 = time.perf_counter()
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    timeline = []
    last = None
    while p.poll() is None:
        try: s = os.path.getsize(out)
        except FileNotFoundError: s = -1
        if s != last:
            timeline.append((round(time.perf_counter()-t0, 2), s))
            last = s
        time.sleep(0.05)
    p.wait()
    end = round(time.perf_counter()-t0, 2)
    # 前 3 个变化点 + 总时长
    print(f"run{run}: first_changes={timeline[:3]} end={end}s final_size={os.path.getsize(out) if os.path.exists(out) else -1}")
