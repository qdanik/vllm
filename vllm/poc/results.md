# H200 - FLASH_ATTN + TRITON Fp8 MoE

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