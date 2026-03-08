```
DOCKER_BUILDKIT=1 docker build . --target vllm-openai --file docker/Dockerfile --target vllm-openai --build-arg max_jobs=48 --build-arg nvcc_threads=12 --build-arg CUDA_VERSION=12.8.1 --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=0.8.1 --build-arg RUN_WHEEL_CHECK=false
```


```
DOCKER_BUILDKIT=1 docker build . --target vllm-openai --file docker/Dockerfile \
  --build-arg max_jobs=48 \
  --build-arg nvcc_threads=12 \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION=0.9.1 \
  --build-arg RUN_WHEEL_CHECK=false \
  --build-arg torch_cuda_arch_list='12.0+PTX' \
   --tag vllm-openai
```