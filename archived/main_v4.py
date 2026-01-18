import os
import subprocess
import re
from funasr import AutoModel

# ================= 配置区域 =================
VIDEO_PATH = "input/4k25.m4v"
OUTPUT_SRT = "output.srt"

# 使用经典的 Paraformer 模型，它兼容性最好
MODEL_ID = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
VAD_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
PUNC_ID = "iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
# ===========================================

def format_timestamp(ms):
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{int(ms):03}"

def clean_text(text):
    text = text.strip()
    # 去除行尾标点
    text = re.sub(r"[。，、？!！.,?~]$", "", text)
    return text

def main():
    temp_wav = "input/4k25_16000.wav"
    
    
    # 显式指定具体的模型 ID，防止 'not registered' 错误
    model = AutoModel(
        model=MODEL_ID,
        vad_model=VAD_ID,
        punc_model=PUNC_ID,
        # vad_kwargs={"max_single_segment_time": 60000}, # 针对长视频优化 VAD
    )

    print("3. 正在识别语音...")
    res = model.generate(
        input=temp_wav,
        batch_size_s=300,
        hotword='',
        merge_vad=True,      # 合并 VAD 片段
        merge_length_s=15,   # 15秒内尝试合并成一句，避免字幕太碎
    )

    print(f"4. 正在写入字幕: {OUTPUT_SRT}...")
    
    # 根据 funasr 版本不同，返回结构可能略有不同
    # 通常结构是 List[Dict]
    if isinstance(res, list) and len(res) > 0:
        # 取结果，Paraformer 通常返回一个 item
        sentence_info = res[0].get('sentence_info', [])
        
        # 兼容性处理：如果 sentence_info 为空，尝试直接解析 text (针对极短音频)
        if not sentence_info and 'text' in res[0]:
             print("警告：未检测到详细时间戳，可能是音频过短或静音。")

        with open(OUTPUT_SRT, "w", encoding="utf-8") as f:
            for i, seg in enumerate(sentence_info):
                # 获取时间戳 (ms)
                # 注意：timestamp 结构通常是 [[start, end], [start, end]...] 或者是 [start, end]
                # 我们取这一句的开始和结束
                ts = seg['timestamp']
                if isinstance(ts[0], list):
                    start_ms = ts[0][0]
                    end_ms = ts[-1][1]
                else:
                    start_ms = ts[0]
                    end_ms = ts[1]
                
                start_str = format_timestamp(start_ms)
                end_str = format_timestamp(end_ms)
                content = clean_text(seg['text'])
                
                f.write(f"{i + 1}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{content}\n\n")
    else:
        print("❌ 识别结果为空，请检查音频文件是否正常。")

    # 清理
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    print("✅ 完成！")

if __name__ == "__main__":
    main()