# Baseline Smoke Profile Summary

- Captured on VM `beast3` with `NVIDIA GeForce RTX 3090`
- Command: `python benchmark/artemis_benchmark.py --model /home/rozhin/rozhin/models/faster-whisper-large-v3 --device cuda --device-index 0 --clip-seconds 30`
- Benchmark mode: `smoke`
- Audio seconds: `30`
- Timed transcription: `2.798s`
- VRAM used: `17180 MiB`

## Top CUDA API time

- `cudaMemcpyAsync`: `34.2%` (`735304519 ns`, `5667` calls)
- `cudaLaunchKernel`: `27.7%` (`595072017 ns`, `148997` calls)
- `cudaStreamSynchronize`: `15.4%` (`331945569 ns`, `3881` calls)
- `cudaMallocAsync_v11020`: `6.0%` (`128448282 ns`, `87459` calls)
- `cuLaunchKernel`: `5.5%` (`117802368 ns`, `29786` calls)

## Top OS runtime time

- `pthread_cond_wait`: `93.5%` (`526265115742 ns`, `196` calls)
- `poll`: `3.3%` (`18297247961 ns`, `192` calls)
- `pthread_cond_timedwait`: `1.6%` (`9002834891 ns`, `19` calls)
- `sem_clockwait`: `0.8%` (`4411063339 ns`, `1` call)
- `futex`: `0.5%` (`2683223835 ns`, `16` calls)
