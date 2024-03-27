# Neural Networks from Scratch

My coursework in second year in HSE AMI.

## Build
```shell
mkdir build
cd build
cmake .. -D NNET_ENABLE_TESTS=ON -D NNET_ENABLE_BENCHMARKS=ON
```

## Test

```shell
cd build
make <TEST_NAME>
./tests/<TEST_NAME>
```
