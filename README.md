## genrate srt file
1. put your videos into `input` folder. Currently supports `.mp4` `.mov` and `.m4v` format
2. run `python generate.py`
3. `.srt` and `.wav` files will be generated into `output` folder in batch

## optimize srt with ai
1. setup openai api key in .env like `OPENAI_API_KEY=xxx`
2. run `python optimize.py`
3. it will handle `.srt` in output folder, and ouput optimized srt and report in md

## use srt file
1. use srt file directly from fcpx: File -> Import -> Captions, select your srt file