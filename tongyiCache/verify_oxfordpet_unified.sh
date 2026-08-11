#!/bin/bash
# Oxford Pet 数据集 - LoRA 方法配置统一性验证脚本

set -e

CONFIG_DIR="config_exps_paper1_PAC/exp1_rebuttel_oxfordPet"
cd "$CONFIG_DIR" || exit 1

echo "=============================================================================="
echo "Oxford Pet 数据集 - LoRA 方法配置统一性验证"
echo "=============================================================================="
echo ""

# 统计 LoRA 方法配置文件
LORA_FILES=$(ls seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
echo "📊 LoRA 方法配置文件总数：$LORA_FILES"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣ 训练参数验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# init_epoch
COUNT=$(grep -l "^init_epoch: 10$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ init_epoch: 10      -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ init_epoch: 10      -> $COUNT/$LORA_FILES ⚠️"
fi

# init_lr
COUNT=$(grep -l "^init_lr: 0.005$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ init_lr: 0.005      -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ init_lr: 0.005      -> $COUNT/$LORA_FILES ⚠️"
fi

# epochs
COUNT=$(grep -l "^epochs: 10$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ epochs: 10          -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ epochs: 10          -> $COUNT/$LORA_FILES ⚠️"
fi

# lrate
COUNT=$(grep -l "^lrate: 0.005$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ lrate: 0.005        -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ lrate: 0.005        -> $COUNT/$LORA_FILES ⚠️"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣ Prefix 标记验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ep10_lr005 标记
COUNT=$(grep -l "ep10_lr005" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ ep10_lr005 标记     -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ ep10_lr005 标记     -> $COUNT/$LORA_FILES ⚠️"
fi

# 检查重复标记
DUPLICATE_COUNT=$(grep "ep10_lr005.*ep10_lr005" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $DUPLICATE_COUNT -eq 0 ]; then
    echo "✅ 无重复标记          -> ✓"
else
    echo "❌ 发现重复标记        -> $DUPLICATE_COUNT 个文件 ⚠️"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣ 按方法分类统计"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

for method in seqlora inclora inflora olora; do
    METHOD_FILES=$(ls ${method}_*.yaml 2>/dev/null | wc -l)
    CONSISTENT=$(grep -l "^init_epoch: 10$" ${method}_*.yaml 2>/dev/null | wc -l)
    
    if [ $CONSISTENT -eq $METHOD_FILES ]; then
        echo "✅ ${method^^} ($METHOD_FILES 个) → 完全一致"
    else
        echo "❌ ${method^^} ($METHOD_FILES 个) → 存在差异"
    fi
done

echo ""
echo "=============================================================================="
echo "示例配置展示 (seqlora_inr_sgd_t20c10_r16.yaml):"
echo "=============================================================================="
head -55 seqlora_inr_sgd_t20c10_r16.yaml | grep -E "^prefix:|^init_epoch:|^init_lr:|^epochs:|^lrate:"
echo ""

echo "=============================================================================="

# 最终判定
ALL_PASS=true

for param in "init_epoch: 10" "init_lr: 0.005" "epochs: 10" "lrate: 0.005"; do
    COUNT=$(grep -l "^${param}$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
    if [ $COUNT -ne $LORA_FILES ]; then
        ALL_PASS=false
        break
    fi
done

PREFIX_COUNT=$(grep -l "ep10_lr005" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $PREFIX_COUNT -ne $LORA_FILES ] || [ $DUPLICATE_COUNT -ne 0 ]; then
    ALL_PASS=false
fi

if [ "$ALL_PASS" = true ]; then
    echo "🎉 ✅ 所有 LoRA 方法配置已完全统一！可以安全用于实验。"
    exit 0
else
    echo "⚠️  ❌ 部分配置未通过验证，请检查上述报告。"
    exit 1
fi
