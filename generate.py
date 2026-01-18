import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from funasr import AutoModel


# ---------------------------
# 基础工具
# ---------------------------

def run_cmd(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()


def ffprobe_duration_sec(media_path: str) -> float:
    out = run_cmd([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        media_path
    ])
    return float(out)


def extract_audio_wav_16k_mono(input_video: str, wav_out: str) -> None:
    # 强制：16k / mono / pcm_s16le，最大限度避免时间漂移
    run_cmd([
        "ffmpeg", "-y",
        "-i", input_video,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        "-c:a", "pcm_s16le",
        wav_out
    ])


def srt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    hh = ms // 3600000
    mm = (ms % 3600000) // 60000
    ss = (ms % 60000) // 1000
    mmm = ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{mmm:03d}"


def norm_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str


def write_srt(segs: List[Segment], out_srt: str) -> None:
    lines = []
    for i, seg in enumerate(segs, 1):
        lines.append(str(i))
        lines.append(f"{srt_time(seg.start_ms)} --> {srt_time(seg.end_ms)}")
        lines.append(seg.text)
        lines.append("")
    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------
# 核心：从 FunASR 输出拿到“句级时间戳”
# ---------------------------

def funasr_to_segments(res: Any, audio_dur_sec: float) -> List[Segment]:
    """
    目标：优先使用 res[0]['sentence_info']，里面每句都有 start/end(ms)
    这是生成 FCPX 可用字幕的最稳方式。
    """
    audio_ms = int(round(audio_dur_sec * 1000))

    if not isinstance(res, list) or len(res) == 0 or not isinstance(res[0], dict):
        raise RuntimeError(f"Unexpected FunASR result format: {type(res)}")

    r0 = res[0]
    sentences = r0.get("sentence_info", None)

    if not sentences:
        # 没 sentence_info 的话，基本就无法保证逐句精准对齐
        # 直接抛错，逼你修参数（比静默生成 1 条字幕靠谱太多）
        raise RuntimeError(
            "FunASR result has NO sentence_info.\n"
            "你需要在 AutoModel 初始化时设置 sentence_timestamp=True。"
        )

    segs: List[Segment] = []
    for s in sentences:
        if not isinstance(s, dict):
            continue
        # text = norm_text(s.get("text", ""))
        text = strip_trailing_punc(norm_text(s.get("text", "")))

        if not text:
            continue

        # FunASR 的 sentence_info 的 start/end 通常是 ms（很多示例都这么用）:contentReference[oaicite:1]{index=1}
        st = int(s.get("start", 0))
        ed = int(s.get("end", 0))

        # 健壮性修正
        if ed <= st:
            ed = st + 400
        if st < 0:
            st = 0
        if ed > audio_ms:
            ed = audio_ms

        segs.append(Segment(st, ed, text))

    # 强制单调递增（避免 FCPX 某些情况下导入异常）
    segs.sort(key=lambda x: (x.start_ms, x.end_ms))
    fixed: List[Segment] = []
    last_end = 0
    for seg in segs:
        if seg.start_ms < last_end:
            seg.start_ms = last_end
        if seg.end_ms <= seg.start_ms:
            seg.end_ms = min(seg.start_ms + 400, audio_ms)
        fixed.append(seg)
        last_end = seg.end_ms

    # 最后一条拉到音频末尾一点，避免“尾巴被截断”
    if fixed and audio_ms - fixed[-1].end_ms >= 800:
        fixed[-1].end_ms = audio_ms

    return fixed

def strip_trailing_punc(text: str) -> str:
    """
    只去掉句尾标点，不影响句中内容。
    例如：
      "你好。" -> "你好"
      "OK!" -> "OK"
      "真的吗？！" -> "真的吗"
      "test..." -> "test"
    """
    if not text:
        return text

    # 句尾可能出现的中英文标点（可按需增删）
    trailing = "，。！？；：、,.!?;:~…—-·\"'）)】]》>》"
    t = text.rstrip()

    # 连续剔除末尾标点
    while t and t[-1] in trailing:
        t = t[:-1].rstrip()

    return t


# ---------------------------
# 主流程
# ---------------------------

def m4v_to_srt(input_video: str, out_srt: str, tmp_dir: str = ".") -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg / ffprobe not found. Please install ffmpeg first.")

    if not os.path.exists(input_video):
        raise FileNotFoundError(input_video)

    base = os.path.splitext(os.path.basename(input_video))[0]
    tmp_wav = os.path.join(tmp_dir, f"{base}__16k.wav")

    # 1) 抽音频（标准化）
    extract_audio_wav_16k_mono(input_video, tmp_wav)

    # 2) 时长
    audio_dur_sec = ffprobe_duration_sec(tmp_wav)

   

    # 3) 关键点：sentence_timestamp=True 让返回结果包含 sentence_info
    model = AutoModel(
        model="paraformer-zh",
        # model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
        vad_model="fsmn-vad",
        punc_model="damo/punc_ct-transformer_cn-en-common-vocab471067-large",
        sentence_timestamp=True,      # ✅ 关键开关：逐句时间戳
        return_raw_text=False
    )

    hotwords = [
        "Action6", "Action5", "DLOGM"
    ]
    res = model.generate(input=tmp_wav, batch_size_s=300, hotword=" ".join(hotwords))

    segments = funasr_to_segments(res, audio_dur_sec)
    write_srt(segments, out_srt)

    print(f"✅ Done: {out_srt}")


if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    
    # 确保 input 和 output 文件夹存在
    if not os.path.exists(input_dir):
        print(f"❌ Error: input directory not found: {input_dir}")
        exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的视频格式
    video_exts = ('.mp4', '.mov', '.m4v')
    
    # 扫描 input 文件夹
    video_files = []
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(video_exts):
            video_files.append(os.path.join(input_dir, fname))
    
    if not video_files:
        print(f"⚠️  No video files found in {input_dir}")
        print(f"   Supported formats: {', '.join(video_exts)}")
        exit(0)
    
    print(f"Found {len(video_files)} video file(s) to process:")
    for vf in video_files:
        print(f"  - {os.path.basename(vf)}")
    print()
    
    # 批量处理
    for video_path in video_files:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_srt = os.path.join(output_dir, f"{base_name}.srt")
        
        print(f"Processing: {os.path.basename(video_path)}...")
        try:
            m4v_to_srt(video_path, out_srt, tmp_dir=output_dir)
        except Exception as e:
            print(f"❌ Failed to process {os.path.basename(video_path)}: {e}")
            continue
    
    print(f"\n🎉 All done! Check {output_dir} for results.")
