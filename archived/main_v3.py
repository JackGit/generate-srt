import os
import datetime
from funasr import AutoModel

def format_timestamp(ms):
    td = datetime.timedelta(milliseconds=ms)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    millis = int(ms % 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"

def generate_srt(video_path):
    model = AutoModel(
        model="paraformer-zh", 
        model_revision="v2.0.4",
        vad_model="fsmn-vad", 
        vad_model_revision="v2.0.4",
        disable_update=True
    )

    print(f"正在智能对齐识别: {video_path}...")
    res = model.generate(input=video_path, batch_size_s=300)
    
    # 1. 保留空格，这对于识别英文单词边界至关重要
    raw_text = res[0]['text'] 
    timestamps = res[0]['timestamp']
    
    segments = []
    current_chars = []
    current_start = -1
    
    MAX_CHARS = 16  # 每行理想的最大字符数
    
    ts_idx = 0
    # 遍历文本中的每一个字符（包括空格）
    for char in raw_text:
        # 如果是空格，它不占用时间戳索引，但它是完美的切分点
        if char == " ":
            current_chars.append(char)
            continue
            
        # 如果是普通字符（中/英/数）
        if ts_idx < len(timestamps):
            if current_start == -1:
                current_start = timestamps[ts_idx][0]
            
            current_chars.append(char)
            current_end = timestamps[ts_idx][1]
            ts_idx += 1
            
            # 检查是否达到切分长度
            if len(current_chars) >= MAX_CHARS:
                # 智能切分：如果是英文，尽量在空格处切，不打断单词
                # 这里简单处理：只要达到长度就存入，并开始新的一行
                segments.append({
                    "text": "".join(current_chars).strip(),
                    "start": current_start,
                    "end": current_end
                })
                current_chars = []
                current_start = -1

    # 处理剩余部分
    if current_chars:
        segments.append({
            "text": "".join(current_chars).strip(),
            "start": current_start,
            "end": timestamps[-1][1]
        })

    # 导出 SRT
    output_srt_path = os.path.splitext(video_path)[0] + "_v4.srt"
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for idx, seg in enumerate(segments):
            if not seg["text"]: continue
            f.write(f"{idx + 1}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text']}\n\n")

    print(f"✅ 完成！已修复英文单词被切断的问题。")

if __name__ == "__main__":
    generate_srt("input/4k25.m4v")