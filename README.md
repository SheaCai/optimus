# Optimus #
[![DOI](https://zenodo.org/badge/364437098.svg)](https://zenodo.org/badge/latestdoi/364437098)
This is the implementation of the paper [Optimus: Towards Optimal Layer-Fusion on Deep Learning Processors]. 

## Getting Started Guide ##
### Start from Source Code ###
1. Create virtual env
```
conda create --name optimusEnv python=3.6
conda activate optimusEnv
```
2. Install requirement
```
pip install -r requirements.txt
```
3. Run a test
```
./test.sh
```
4. To find out all the options
```
python ./fusion/tools/optimal_schedule_search.py --help
```

### Experiments ###
1. Run overall experiment to get the memory access and energy over multiple models
  The result will be stored in result/overall_experiment/. It will take more than ten minutes to complete this experiment.
```
python ./fusion/experiment/overall_experiment.py
```

2. Run memory access analysis over multiple models
  The result will be stored in result/analysis/. It will take more than ten minutes to complete this experiment.
```
python ./fusion/experiment/analysis.py
```

3. Evaluate the Impact of Batch Size
  The result will be stored in result/batch_size/.
```
python ./fusion/experiment/batch_size.py
```

4. Evaluate the impact of on-chip memory space
  The result will be stored in result/buffer_size/.
```
python ./fusion/experiment/buffer_size.py
```

5. Evaluate the impact of Dataflow
  This experiment supports the experiment results of section 4.2.5 in our paper, and  the result will be stored  in result/dataflow/.
```
python ./fusion/experiment/dataflow.py
```

6. Evaluate the impact of PE-array and buffer
  The result will be stored in result/pe_array/.
```
python ./fusion/experiment/pe_array.py
```

7. Evaluate the performance on different processors
  The result will be stored in result/processor/.
```
python ./fusion/experiment/processor.py
```
