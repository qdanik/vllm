# H200 - FLASH_ATTN + TRITON Fp8 MoE

Calculating optimal batch_size...
  Calculated batch_size: 65
    seq_len=1024, hidden_size=4096, num_layers=94
    total_memory=139.8GB
    reserved_memory=0.0GB (includes KV cache)
    allocated_memory=0.0GB
    free_memory=97.9GB (usable)
    mem_per_sample=1520.0MB

Running 10 batches with batch_size=65...

Warmup run...

Profiling...
  Run  1: 3702.0ms, 65 nonces (17.6/sec, 56.95ms/nonce)

Validation result:
  n_total=65, n_mismatch=0, p_value=1.000000, fraud_detected=False
  Run  2: 3699.6ms, 65 nonces (17.6/sec, 56.92ms/nonce)
  Run  3: 3696.5ms, 65 nonces (17.6/sec, 56.87ms/nonce)
  Run  4: 3697.3ms, 65 nonces (17.6/sec, 56.88ms/nonce)
  Run  5: 3697.3ms, 65 nonces (17.6/sec, 56.88ms/nonce)
  Run  6: 3699.8ms, 65 nonces (17.6/sec, 56.92ms/nonce)
  Run  7: 3702.6ms, 65 nonces (17.6/sec, 56.96ms/nonce)
  Run  8: 3701.8ms, 65 nonces (17.6/sec, 56.95ms/nonce)
  Run  9: 3698.9ms, 65 nonces (17.6/sec, 56.91ms/nonce)
  Run 10: 3700.8ms, 65 nonces (17.6/sec, 56.93ms/nonce)

======================================================================
RESULTS:
======================================================================
Batch size used: 65
Total batches: 10
Total nonces: 650
Average batch time: 3699.7ms
Average rate: 17.57 nonces/sec
Average rate: 1054 nonces/min
Time per nonce: 56.92ms

Batch time variance: 3696.5 - 3702.6ms (±0.2%)
  Variant 1: 7bZptns1KbJ6LigylrOttJe1ILY7MsKp
  Variant 2: CDNAOTip46hdMU21CLSjLa+2EK55rm81
  Variant 3: FbBQOLOwzLMttr43krWAp+wtty1ntLqv
  Variant 4: t7Z4r+c0WK4bNSgtpTAmMWs0bLJptYg4
  Variant 5: ajAysb21erSuNjs0yK5us62w8rNtt/01
  Variant 6: T6oFJx+wkbGJuiqyF7WBoCAkFTTks0mw
  Variant 7: TjXGM4cs+DB5t+OtTDYNLCC21zG6N2Kr
  Variant 8: JzF6rRq1cbdvMlEzXjNHsEexKR4cuAw3
  Variant 9: +rNUsW61kbQPNEc1gzQjsmU2LZhQNkY1
  Variant 10: CC0sMlUx1C4ztnIiY7dTuZAtizRlMOSr
  ✓ Hash matches production baseline

======================================================================
Performance vs target:
  Current: 56.92ms/nonce
  Target:  4.80ms/nonce
  Gap: 1085.8% slower (need 1086% improvement)

Estimated time for 1000 nonces: 56.9s (0.9min)
Estimated time for 10000 nonces: 569.2s (9.5min)

# H200 - FLASH_ATTN + DEEPGEMM Fp8 MoE

Calculating optimal batch_size...
  Calculated batch_size: 65
    seq_len=1024, hidden_size=4096, num_layers=94
    total_memory=139.8GB
    reserved_memory=0.0GB (includes KV cache)
    allocated_memory=0.0GB
    free_memory=97.9GB (usable)
    mem_per_sample=1520.0MB

Running 10 batches with batch_size=65...

Warmup run...

Profiling...
  Run  1: 3702.0ms, 65 nonces (17.6/sec, 56.95ms/nonce)

Validation result:
  n_total=65, n_mismatch=0, p_value=1.000000, fraud_detected=False
  Run  2: 3699.6ms, 65 nonces (17.6/sec, 56.92ms/nonce)
  Run  3: 3696.5ms, 65 nonces (17.6/sec, 56.87ms/nonce)
  Run  4: 3697.3ms, 65 nonces (17.6/sec, 56.88ms/nonce)
  Run  5: 3697.3ms, 65 nonces (17.6/sec, 56.88ms/nonce)
  Run  6: 3699.8ms, 65 nonces (17.6/sec, 56.92ms/nonce)
  Run  7: 3702.6ms, 65 nonces (17.6/sec, 56.96ms/nonce)
  Run  8: 3701.8ms, 65 nonces (17.6/sec, 56.95ms/nonce)
  Run  9: 3698.9ms, 65 nonces (17.6/sec, 56.91ms/nonce)
  Run 10: 3700.8ms, 65 nonces (17.6/sec, 56.93ms/nonce)

======================================================================
RESULTS:
======================================================================
Batch size used: 65
Total batches: 10
Total nonces: 650
Average batch time: 3699.7ms
Average rate: 17.57 nonces/sec
Average rate: 1054 nonces/min
Time per nonce: 56.92ms

Batch time variance: 3696.5 - 3702.6ms (±0.2%)
  Variant 1: 7bZptns1KbJ6LigylrOttJe1ILY7MsKp
  Variant 2: CDNAOTip46hdMU21CLSjLa+2EK55rm81
  Variant 3: FbBQOLOwzLMttr43krWAp+wtty1ntLqv
  Variant 4: t7Z4r+c0WK4bNSgtpTAmMWs0bLJptYg4
  Variant 5: ajAysb21erSuNjs0yK5us62w8rNtt/01
  Variant 6: T6oFJx+wkbGJuiqyF7WBoCAkFTTks0mw
  Variant 7: TjXGM4cs+DB5t+OtTDYNLCC21zG6N2Kr
  Variant 8: JzF6rRq1cbdvMlEzXjNHsEexKR4cuAw3
  Variant 9: +rNUsW61kbQPNEc1gzQjsmU2LZhQNkY1
  Variant 10: CC0sMlUx1C4ztnIiY7dTuZAtizRlMOSr
  ✓ Hash matches production baseline

======================================================================
Performance vs target:
  Current: 56.92ms/nonce
  Target:  4.80ms/nonce
  Gap: 1085.8% slower (need 1086% improvement)

Estimated time for 1000 nonces: 56.9s (0.9min)
Estimated time for 10000 nonces: 569.2s (9.5min)

# H100 - FLASH_ATTN + TRITON Fp8 MoE

Calculating optimal batch_size...
  Calculated batch_size: 37
    seq_len=1024, hidden_size=4096, num_layers=94
    total_memory=79.2GB
    reserved_memory=0.0GB (includes KV cache)
    allocated_memory=0.0GB
    free_memory=55.4GB (usable)
    mem_per_sample=1520.0MB

Running 10 batches with batch_size=37...

Warmup run...

Profiling...
  Run  1: 2317.5ms, 37 nonces (16.0/sec, 62.63ms/nonce)
  Run  2: 2316.1ms, 37 nonces (16.0/sec, 62.60ms/nonce)
  Run  3: 2316.5ms, 37 nonces (16.0/sec, 62.61ms/nonce)
  Run  4: 2316.9ms, 37 nonces (16.0/sec, 62.62ms/nonce)
  Run  5: 2316.9ms, 37 nonces (16.0/sec, 62.62ms/nonce)
  Run  6: 2317.1ms, 37 nonces (16.0/sec, 62.62ms/nonce)
  Run  7: 2316.8ms, 37 nonces (16.0/sec, 62.62ms/nonce)
  Run  8: 2317.1ms, 37 nonces (16.0/sec, 62.62ms/nonce)
  Run  9: 2316.8ms, 37 nonces (16.0/sec, 62.62ms/nonce)
  Run 10: 2317.5ms, 37 nonces (16.0/sec, 62.64ms/nonce)

======================================================================
RESULTS:
======================================================================
Batch size used: 37
Total batches: 10
Total nonces: 370
Average batch time: 2316.9ms
Average rate: 15.97 nonces/sec
Average rate: 958 nonces/min
Time per nonce: 62.62ms

Batch time variance: 2316.1 - 2317.5ms (±0.1%)
  Variant 1: jyxktD4wZ7FgsbE1rLgTMsevZxw7trO2
  Variant 2: Z7dNtoM16rE3LpIy4LKytGC1yrUQMwyp
  Variant 3: eTOztLWzHTBgsMU3k7FWuEgskbJlNsuk
  Variant 4: GSlqOXC2Cy5vsVI21bFxK2y0Va+ns8oo
  Variant 5: ybTPqPY1uzGjJTuyVLmErz2yQrY7qhez
  Variant 6: 8i7vNH8yHrRirUcsRC+0rKKxczenuRgu
  Variant 7: Zra6pnqldDi8sxi007Vgsj62NLTQrtEk
  Variant 8: qrIEOHuo8qvnKv6vJrgBLQ41E7jIMtqp
  Variant 9: +S08tOazYaDCuDA2KrUatAW1tS9WHl+0
  Variant 10: Ci2qMuc3qDGusfk12jeLq9W16bAFML80
  ✗ Hash MISMATCH from production!
    Expected: 7bZptns1KbJ6LigylrOttJe1ILY7MsKp
    Got:      Z7dNtoM16rE3LpIy4LKytGC1yrUQMwyp

======================================================================
Performance vs target:
  Current: 62.62ms/nonce
  Target:  4.80ms/nonce
  Gap: 1204.6% slower (need 1205% improvement)

Estimated time for 1000 nonces: 62.6s (1.0min)
Estimated time for 10000 nonces: 626.2s (10.4min)

# H100 - FLASH_ATTN + FLASHINFER_CUTLASS

Batch size used: 37
Total batches: 10
Total nonces: 370
Average batch time: 3010.1ms
Average rate: 12.29 nonces/sec
Average rate: 738 nonces/min
Time per nonce: 81.35ms