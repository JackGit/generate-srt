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
        punc_model="ct-punc-c", 
        punc_model_revision="v2.0.4",
        disable_update=True
    )

    print(f"正在分析视频: {video_path}...")
    # 修改 generate 这一行
    res = model.generate(
        input=video_path, 
        batch_size_s=300,
        hotword='Action6', # 顺便优化下你的 Action6 识别
        # 强制要求返回 timestamp 且保持对齐
        return_spk=False, 
        output_dir=None
    )

    # 打印一下看看结构
    print(res[0].keys())
    
    if not res or 'timestamp' not in res[0]:
        print("错误：未能获取时间戳信息")
        return

    raw_text = res[0]['text']
    timestamps = res[0]['timestamp'] # 字符级时间戳列表 [[s,e], [s,e]...]
    
    # --- 核心合并逻辑 ---
    segments = []
    current_sentence = ""
    start_time = -1
    
    # 标点符号用于切分句子
    punc_list = {"。", "？", "！", "，", "；", "?", "!", ","}
    
    ts_idx = 0
    for char in raw_text:
        # 如果是空格，跳过
        if char == " ":
            continue
            
        # 如果是标点符号
        if char in punc_list:
            if current_sentence:
                # 结束当前句子
                segments.append({
                    "text": current_sentence + char,
                    "start": start_time,
                    "end": timestamps[ts_idx-1][1] if ts_idx > 0 else 0
                })
                current_sentence = ""
                start_time = -1
            continue

        # 如果是普通字符
        if ts_idx < len(timestamps):
            if start_time == -1:
                start_time = timestamps[ts_idx][0]
            current_sentence += char
            
            # 如果句子太长（超过20个字），即使没标点也强制换行，防止字幕铺满屏幕
            if len(current_sentence) > 20:
                segments.append({
                    "text": current_sentence,
                    "start": start_time,
                    "end": timestamps[ts_idx][1]
                })
                current_sentence = ""
                start_time = -1
            
            ts_idx += 1

    # 处理最后剩余的内容
    if current_sentence:
        segments.append({
            "text": current_sentence,
            "start": start_time,
            "end": timestamps[-1][1]
        })

    # --- 写入 SRT 文件 ---
    output_srt_path = os.path.splitext(video_path)[0] + ".srt"
    with open(output_srt_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments):
            # 过滤掉无效的时间戳
            if seg["start"] == -1: continue
            
            f.write(f"{i + 1}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"{seg['text'].strip()}\n\n")

    print(f"✅ 完成！SRT 已生成: {output_srt_path}")

if __name__ == "__main__":
    generate_srt("input/4k25.m4v")