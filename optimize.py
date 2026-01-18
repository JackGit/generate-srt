import os
import re
from datetime import datetime
from typing import List, Tuple
from dotenv import load_dotenv  # 从 python-dotenv 导入 load_dotenv
load_dotenv()  # 加载 .env 文件中的环境变量

from openai import OpenAI

api_key = os.environ.get("OPENAI_API_KEY", None)
openai_model = "gpt-5.1"


def parse_srt(srt_path: str) -> List[Tuple[int, str, str, str]]:
    """
    解析 SRT 文件
    返回: [(序号, 时间轴, 原文本, 原文本), ...]
    """
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 分割每个字幕块
    blocks = re.split(r'\n\n+', content.strip())
    subtitles = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            index = lines[0].strip()
            timestamp = lines[1].strip()
            text = '\n'.join(lines[2:]).strip()
            subtitles.append((index, timestamp, text, text))
    
    return subtitles


def build_prompt(subtitles: List[Tuple[int, str, str, str]]) -> str:
    """
    构建发送给 OpenAI 的 prompt
    """
    # 提取所有文本
    texts = [f"{idx}. {text}" for idx, timestamp, text, _ in subtitles]
    text_content = "\n".join(texts)
    
    prompt = f"""你是一个专业的字幕校对助手。以下是从语音识别(ASR)生成的字幕文本，请帮我校对。
        **重要原则：**
        1. **只修正明显的错误**：错别字、同音字错误、明显的语法错误
        2. **数字和英文规范化**：
        - 根据上下文判断是用阿拉伯数字(1,2,3)、中文数字(一、二、三)还是罗马数字(I, II, III)
        - 英文专有名词、品牌名、技术术语等要正确大小写
        3. **保持原意**：不要改写句子、不要添加内容、不要修改说话风格
        4. **避免过度修改**：因为这是 ASR 字幕，修改太多会与实际语音不匹配

        请返回 JSON 格式：
        ```json
        {{
        "corrections": [
            {{"index": "1", "original": "原文", "corrected": "修正后", "reason": "修正原因"}},
            {{"index": "3", "original": "原文", "corrected": "修正后", "reason": "修正原因"}}
        ],
        "summary": "本次校对的总体说明（修正了几处，主要问题是什么）"
        }}
        ```

        如果某条字幕无需修改，就不要在 corrections 里出现。

        ---

        字幕内容：

        {text_content}
     """
    
    return prompt


def call_openai(prompt: str, api_key: str = None, model: str = "gpt-5-nano") -> str:
    """
    调用 OpenAI API
    """
    
    client = OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个专业的字幕校对助手，擅长发现并修正 ASR 字幕中的错误。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content


def apply_corrections(subtitles: List[Tuple[int, str, str, str]], corrections: List[dict]) -> List[Tuple[int, str, str, str]]:
    """
    应用修正到字幕
    返回: [(序号, 时间轴, 原文本, 修正后文本), ...]
    """
    # 建立索引映射
    correction_map = {c['index']: c for c in corrections}
    
    result = []
    for idx, timestamp, original, _ in subtitles:
        if idx in correction_map:
            corrected = correction_map[idx]['corrected']
            result.append((idx, timestamp, original, corrected))
        else:
            result.append((idx, timestamp, original, original))
    
    return result


def write_optimized_srt(subtitles: List[Tuple[int, str, str, str]], output_path: str) -> None:
    """
    写入优化后的 SRT 文件
    """
    lines = []
    for idx, timestamp, _, corrected in subtitles:
        lines.append(idx)
        lines.append(timestamp)
        lines.append(corrected)
        lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def write_report(
    input_srt: str,
    output_srt: str,
    subtitles: List[Tuple[int, str, str, str]],
    corrections: List[dict],
    summary: str,
    report_path: str
) -> None:
    """
    生成校对报告 Markdown 文件
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        f"# 字幕校对报告",
        f"",
        f"**生成时间**: {timestamp}  ",
        f"**原始文件**: `{os.path.basename(input_srt)}`  ",
        f"**优化文件**: `{os.path.basename(output_srt)}`  ",
        f"",
        f"## 总体说明",
        f"",
        f"{summary}",
        f"",
        f"## 修正详情",
        f"",
    ]
    
    if not corrections:
        lines.append("✅ 未发现需要修正的问题。")
    else:
        lines.append(f"共修正 **{len(corrections)}** 处：")
        lines.append("")
        
        for c in corrections:
            idx = c['index']
            original = c['original']
            corrected = c['corrected']
            reason = c['reason']
            
            lines.append(f"### #{idx}")
            lines.append(f"")
            lines.append(f"- **原文**: {original}")
            lines.append(f"- **修正**: {corrected}")
            lines.append(f"- **原因**: {reason}")
            lines.append(f"")
    
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 完整对照")
    lines.append("")
    lines.append("| 序号 | 原文 | 修正后 | 状态 |")
    lines.append("|------|------|--------|------|")
    
    for idx, _, original, corrected in subtitles:
        status = "✏️" if original != corrected else "✓"
        original_short = original[:50] + "..." if len(original) > 50 else original
        corrected_short = corrected[:50] + "..." if len(corrected) > 50 else corrected
        lines.append(f"| {idx} | {original_short} | {corrected_short} | {status} |")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def optimize_srt(input_srt: str, output_dir: str, api_key: str = None, model: str = "gpt-5-nano") -> None:
    """
    主流程：优化 SRT 字幕
    """
    if not os.path.exists(input_srt):
        raise FileNotFoundError(f"Input SRT file not found: {input_srt}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(input_srt))[0]
    output_srt = os.path.join(output_dir, f"{base_name}_optimized.srt")
    report_path = os.path.join(output_dir, f"{base_name}_report.md")
    
    print(f"📖 Reading: {os.path.basename(input_srt)}")
    subtitles = parse_srt(input_srt)
    print(f"   Found {len(subtitles)} subtitle entries")
    
    print(f"🤖 Calling OpenAI API ({model})...")
    prompt = build_prompt(subtitles)
    response_text = call_openai(prompt, api_key, model)
    
    # 解析 JSON 响应
    import json
    try:
        response_data = json.loads(response_text)
        corrections = response_data.get('corrections', [])
        summary = response_data.get('summary', '无总结说明')
    except json.JSONDecodeError as e:
        print(f"⚠️  Failed to parse OpenAI response: {e}")
        print(f"Response: {response_text}")
        corrections = []
        summary = "解析失败"
    
    print(f"   Found {len(corrections)} corrections")
    
    # 应用修正
    optimized_subtitles = apply_corrections(subtitles, corrections)
    
    # 写入优化后的 SRT
    write_optimized_srt(optimized_subtitles, output_srt)
    print(f"✅ Saved: {output_srt}")
    
    # 生成报告
    write_report(input_srt, output_srt, optimized_subtitles, corrections, summary, report_path)
    print(f"📝 Report: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 OpenAI 优化 ASR 生成的字幕")
    parser.add_argument("input", nargs="?", help="输入 .srt 文件路径（可选，默认处理 output 文件夹下所有 srt）")
    parser.add_argument("--output", default="output", help="输出目录（默认: output）")
    parser.add_argument("--all", action="store_true", help="处理 output 文件夹下所有 srt 文件（排除已优化的）")
    
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, args.output)
    
    # 确定要处理的文件
    srt_files = []
    
    if args.input:
        # 处理指定文件
        srt_files.append(args.input)
    elif args.all:
        # 处理 output 文件夹下所有 srt（排除 _optimized.srt）
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if fname.endswith('.srt') and not fname.endswith('_optimized.srt'):
                    srt_files.append(os.path.join(output_dir, fname))
    else:
        # 默认：处理 output 文件夹下所有 srt（排除 _optimized.srt）
        if os.path.exists(output_dir):
            for fname in os.listdir(output_dir):
                if fname.endswith('.srt') and not fname.endswith('_optimized.srt'):
                    srt_files.append(os.path.join(output_dir, fname))
    
    if not srt_files:
        print("⚠️  No SRT files found to process.")
        print(f"   Searched in: {output_dir}")
        print("   Use --help for usage information.")
        exit(0)
    
    print(f"Found {len(srt_files)} SRT file(s) to optimize:")
    for srt in srt_files:
        print(f"  - {os.path.basename(srt)}")
    print()
    
    # 批量处理
    for srt_path in srt_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(srt_path)}")
        print('='*60)
        
        try:
            optimize_srt(srt_path, output_dir, api_key, openai_model)
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(srt_path)}: {e}")
            continue
    
    print(f"\n🎉 All done! Check {output_dir} for results.")
