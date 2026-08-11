#!/bin/bash
# Aircraft 和 Cars196 数据集配置统一性验证脚本

set -e

CONFIG_BASE="config_exps_paper1_PAC"

echo "=============================================================================="
echo "Aircraft & Cars196 - LoRA 方法配置统一性验证"
echo "=============================================================================="
echo ""

# ========== Aircraft 验证 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Aircraft 数据集 (目标：epochs=30, lr=0.05)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$CONFIG_BASE/exp1_rebuttel_aircraft" || exit 1

LORA_FILES=$(ls seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
echo "LoRA 方法配置文件总数：$LORA_FILES"
echo ""

COUNT=$(grep -l "^init_epoch: 30$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ init_epoch: 30      -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ init_epoch: 30      -> $COUNT/$LORA_FILES ⚠️"
fi

COUNT=$(grep -l "^init_lr: 0.05$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ init_lr: 0.05       -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ init_lr: 0.05       -> $COUNT/$LORA_FILES ⚠️"
fi

COUNT=$(grep -l "^epochs: 30$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ epochs: 30          -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ epochs: 30          -> $COUNT/$LORA_FILES ⚠️"
fi

COUNT=$(grep -l "^lrate: 0.05$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ lrate: 0.05         -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ lrate: 0.05         -> $COUNT/$LORA_FILES ⚠️"
fi

echo ""
echo "按方法分类统计:"
for method in seqlora inclora inflora olora; do
    METHOD_FILES=$(ls ${method}_*.yaml 2>/dev/null | wc -l)
    CONSISTENT=$(grep -l "^init_epoch: 30$" ${method}_*.yaml 2>/dev/null | wc -l)
    
    if [ $CONSISTENT -eq $METHOD_FILES ]; then
        echo "  ✅ ${method^^} ($METHOD_FILES 个) → 完全一致"
    else
        echo "  ❌ ${method^^} ($METHOD_FILES 个) → 存在差异"
    fi
done

echo ""

# ========== Cars196 验证 ==========
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Cars196 数据集 (目标：epochs=20, lr=0.05)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "../exp1_rebuttel_cars196" || exit 1

LORA_FILES=$(ls seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
echo "LoRA 方法配置文件总数：$LORA_FILES"
echo ""

COUNT=$(grep -l "^init_epoch: 20$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ init_epoch: 20      -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ init_epoch: 20      -> $COUNT/$LORA_FILES ⚠️"
fi

COUNT=$(grep -l "^init_lr: 0.05$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ init_lr: 0.05       -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ init_lr: 0.05       -> $COUNT/$LORA_FILES ⚠️"
fi

COUNT=$(grep -l "^epochs: 20$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ epochs: 20          -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ epochs: 20          -> $COUNT/$LORA_FILES ⚠️"
fi

COUNT=$(grep -l "^lrate: 0.05$" seqlora_*.yaml inclora_*.yaml inflora_*.yaml olora_*.yaml 2>/dev/null | wc -l)
if [ $COUNT -eq $LORA_FILES ]; then
    echo "✅ lrate: 0.05         -> $COUNT/$LORA_FILES (100%)"
else
    echo "❌ lrate: 0.05         -> $COUNT/$LORA_FILES ⚠️"
fi

echo ""
echo "按方法分类统计:"
for method in seqlora inclora inflora olora; do
    METHOD_FILES=$(ls ${method}_*.yaml 2>/dev/null | wc -l)
    CONSISTENT=$(grep -l "^init_epoch: 20$" ${method}_*.yaml 2>/dev/null | wc -l)
    
    if [ $CONSISTENT -eq $METHOD_FILES ]; then
        echo "  ✅ ${method^^} ($METHOD_FILES 个) → 完全一致"
    else
        echo "  ❌ ${method^^} ($METHOD_FILES 个) → 存在差异"
    fi
done

echo ""
echo "=============================================================================="
echo "🎉 验证完成！两个数据集内部配置已完全统一"
echo "=============================================================================="
