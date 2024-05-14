# Neural Networks from Scratch

My coursework in second year in HSE AMI.

## Build
```shell
mkdir build
cd build
cmake .. -D NNET_ENABLE_TESTS=ON
```

## Test

test correctness

```shell
cd build
make <TEST_NAME>
./tests/<TEST_NAME>
```

execute python scripts in the notebook and run tests which trains on data produced

```shell
cd build
make run
./run
```
