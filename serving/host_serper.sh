

export PYTHONPATH=/data/home/fractal/shreyas/serving/web_agents
export JINA_API_KEY=jina_xx
export SERPER_API_KEY=xx
export OPENAI_API_KEY=sk-xx
export MAX_OUTBOUND=256
export JINA_CACHE_DIR=/data/home/fractal/shreyas/ReCall/scripts/serving/.cache/jina_cache
export SERPER_CACHE_DIR=/data/home/fractal/shreyas/ReCall/scripts/serving/.cache/serper_cache
python3 sandbox_serper.py --port 1211 --workers  128 
