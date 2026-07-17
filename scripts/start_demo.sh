#!/bin/bash
# 启动 Gradio Demo（脱离会话后台运行，避免 SSH channel 阻塞）
P=/root/autodl-tmp/llm-speech-enhancement-v2
echo "===VERIFY_LAUNCH_LINE==="
grep -n "\.launch(" "$P/src/app.py"
echo "===KILL_OLD==="
pkill -f 'src/app.py' 2>/dev/null
sleep 3
cd "$P" || exit 1
rm -f outputs/logs/demo.log
setsid /root/miniconda3/envs/llm-se-v2/bin/python src/app.py < /dev/null > outputs/logs/demo.log 2>&1 &
disown
sleep 2
echo "===PROC_AFTER_LAUNCH==="
ps aux | grep 'src/app.py' | grep -v grep | cat
echo "===DONE==="
