#!/bin/bash
# Quick backend test matrix for PoC performance
# Tests different attention and MoE backends

set -e

ATTN_BACKENDS=("FLASH_ATTN" "FLASHINFER")
MOE_BACKENDS=("TRITON" "AITER" "MARLIN")

EXPECTED_HASH="ZbWht/cxNDSFMHo0a7CvtDOztLVpNp4v"

echo "========================================================================"
echo "Backend Matrix Test - Quick Mode"
echo "========================================================================"
echo ""

declare -A results

for attn in "${ATTN_BACKENDS[@]}"; do
    for moe in "${MOE_BACKENDS[@]}"; do
        echo "----------------------------------------"
        echo "Testing: ATTN=${attn}, MoE=${moe}"
        echo "----------------------------------------"
        
        export VLLM_USE_V1=1
        export TEST_LARGE_MODEL=1
        export VLLM_ATTENTION_BACKEND=${attn}
        export VLLM_FP8_MOE_BACKEND=${moe}
        
        # Run test and capture output
        if output=$(python3 test_qwen3_poc.py 2>&1); then
            # Parse throughput
            throughput=$(echo "$output" | grep "Throughput:" | grep -oE '[0-9]+\.[0-9]+' | head -1)
            
            # Parse latency
            latency=$(echo "$output" | grep "Average:" | grep -oE '\([0-9]+\.[0-9]+ms per nonce\)' | grep -oE '[0-9]+\.[0-9]+' | head -1)
            
            # Parse hash
            hash=$(echo "$output" | grep "First nonce hash:" | awk '{print $NF}')
            
            # Check hash
            if [ "$hash" = "$EXPECTED_HASH" ]; then
                hash_status="✓"
                results["${attn}_${moe}"]="${latency}ms | ${throughput}/s | MATCH"
                echo "✓ Hash MATCH - Latency: ${latency}ms, Throughput: ${throughput}/s"
            else
                hash_status="✗"
                results["${attn}_${moe}"]="${latency}ms | ${throughput}/s | WRONG"
                echo "✗ Hash MISMATCH - Expected: ${EXPECTED_HASH}, Got: ${hash}"
            fi
        else
            echo "✗ FAILED to run test"
            results["${attn}_${moe}"]="FAILED"
        fi
        
        echo ""
    done
done

echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
printf "%-20s %-40s\n" "Configuration" "Result"
echo "------------------------------------------------------------------------"

for key in "${!results[@]}"; do
    attn=$(echo "$key" | cut -d'_' -f1)
    moe=$(echo "$key" | cut -d'_' -f2)
    result="${results[$key]}"
    printf "%-20s %-40s\n" "${attn}/${moe}" "${result}"
done

echo "========================================================================"
