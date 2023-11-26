# MuSHRoom method.
1. Environment Installation

Please set up a new virtual environment and install the following package:
The repository is based on ([sdftudio](https://github.com/autonomousvision/sdfstudio)). Please install sdfstudio first.

2. Please first ender the mushroom_method folder as the root.
To get the Nerfecto results evaluated by testing within a single sequence, please train with
```
./train_nerfacto_within_same.sh 
```

To get the Nerfecto results evaluated by testing with a different sequence, please train with
```
./train_nerfacto_with_diff.sh 
```


To get the result of our method evaluated by testing within a single sequence, please train with
```
./train_our_within_same.sh 
```

To get the result of our method evaluated by testing with a different sequence, please train with
```
./train_our_with_diff.sh 
```

Evaluation can be done by 
```
ns-eval —load-config="path/config.yml" —output-path="path/output.json"
```