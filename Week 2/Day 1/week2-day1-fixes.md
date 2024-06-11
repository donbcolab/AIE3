export PYTHONPATH="${PYTHONPATH}:/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace"


export PYTHONPATH="/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace":$PYTHONPATH


export PYTHONPATH="/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace":/home/donbr/miniconda3/envs/llmops-course/lib/python3.11/site-packages


python
>>> import sys
>>> print(sys.path)
>>> from aimakerspace.openai_utils.embedding import EmbeddingModel

find "/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/.venv/bin/python" -name "site-packages"

find /home/donbr/miniconda3/envs/llmops-course/ -name "site-packages"

find "/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/.venv/" -name "site-packages"


(.venv) donbr@lapdog:~/aie3-bootcamp/AIE3/Week 2/Day 1$ python "/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace/vectordatabase.py"
Traceback (most recent call last):
  File "/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace/vectordatabase.py", line 4, in <module>
    from aimakerspace.openai_utils.embedding import EmbeddingModel
ModuleNotFoundError: No module named 'aimakerspace'
(.venv) donbr@lapdog:~/aie3-bootcamp/AIE3/Week 2/Day 1$ echo $PYTHONPATH
/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace:/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/.venv/lib/python3.11/site-packages
(.venv) donbr@lapdog:~/aie3-bootcamp/AIE3/Week 2/Day 1$ python
Python 3.11.8 (main, Mar  9 2024, 21:58:26) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import sys
>>> print(sys.path)
['', '/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace', '/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/.venv/lib/python3.11/site-packages', '/home/donbr/.pyenv/versions/3.11.8/lib/python311.zip', '/home/donbr/.pyenv/versions/3.11.8/lib/python3.11', '/home/donbr/.pyenv/versions/3.11.8/lib/python3.11/lib-dynload']
>>> from aimakerspace.openai_utils.embedding import EmbeddingModel
>>> exit()

    PYTHONPATH="/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace:/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/.venv/lib/python3.11/site-packages" python "/home/donbr/aie3-bootcamp/AIE3/Week 2/Day 1/aimakerspace/vectordatabase.py"
