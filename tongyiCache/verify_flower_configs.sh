#!/bin/bash
# Flowers 数据集配置验证脚本
# 用途：验证所有配置文件已正确更新训练参数和 prefix 标记

set -e

CONFIG_DIR="config_exps_paper1_PAC/exp1_rebuttel_flower"
cd "$CONFIG_DIR" || exit 1

echo "=============================================================================="
echo "Flowers 数据集配置验证报告"
echo "=============================================================================="
echo ""

TOTAL_FILES=$(ls *.yaml | wc -l)
echo "📁 配置文件目录：$CONFIG_DIR"
echo "📊 总配置文件数：$TOTAL_FILES"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣ 训练参数验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# init_epoch
COUNT=$(grep -c "^init_epoch: 10$" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
PERCENT=$((COUNT * 100 / TOTAL_FILES))
if [ $COUNT -eq $TOTAL_FILES ]; then
    echo "✅ init_epoch: 10      -> $COUNT/$TOTAL_FILES ($PERCENT%)"
else
    echo "❌ init_epoch: 10      -> $COUNT/$TOTAL_FILES ($PERCENT%) ⚠️"
fi

# init_lr
COUNT=$(grep -c "^init_lr: 0.005$" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
PERCENT=$((COUNT * 100 / TOTAL_FILES))
if [ $COUNT -eq $TOTAL_FILES ]; then
    echo "✅ init_lr: 0.005      -> $COUNT/$TOTAL_FILES ($PERCENT%)"
else
    echo "❌ init_lr: 0.005      -> $COUNT/$TOTAL_FILES ($PERCENT%) ⚠️"
fi

# epochs (注意避免匹配 init_epoch)
COUNT=$(grep -c "^epochs: 10$" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
PERCENT=$((COUNT * 100 / TOTAL_FILES))
if [ $COUNT -eq $TOTAL_FILES ]; then
    echo "✅ epochs: 10          -> $COUNT/$TOTAL_FILES ($PERCENT%)"
else
    echo "❌ epochs: 10          -> $COUNT/$TOTAL_FILES ($PERCENT%) ⚠️"
fi

# lrate
COUNT=$(grep -c "^lrate: 0.005$" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
PERCENT=$((COUNT * 100 / TOTAL_FILES))
if [ $COUNT -eq $TOTAL_FILES ]; then
    echo "✅ lrate: 0.005        -> $COUNT/$TOTAL_FILES ($PERCENT%)"
else
    echo "❌ lrate: 0.005        -> $COUNT/$TOTAL_FILES ($PERCENT%) ⚠️"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣ Prefix 标记验证"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ep10_lr005 标记
COUNT=$(grep -c "ep10_lr005" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
PERCENT=$((COUNT * 100 / TOTAL_FILES))
if [ $COUNT -eq $TOTAL_FILES ]; then
    echo "✅ ep10_lr005 标记     -> $COUNT/$TOTAL_FILES ($PERCENT%)"
else
    echo "❌ ep10_lr005 标记     -> $COUNT/$TOTAL_FILES ($PERCENT%) ⚠️"
fi

# 检查重复标记
DUPLICATE_COUNT=$(grep "ep10_lr005.*ep10_lr005" *.yaml 2>/dev/null | wc -l)
if [ $DUPLICATE_COUNT -eq 0 ]; then
    echo "✅ 无重复标记          -> ✓"
else
    echo "❌ 发现重复标记        -> $DUPLICATE_COUNT 个文件 ⚠️"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣ 示例配置展示"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📄 seqlora_inr_sgd_t20c10_r16.yaml:"
echo "----------------------------------------"
head -55 seqlora_inr_sgd_t20c10_r16.yaml | grep -E "^prefix:|^init_epoch:|^init_lr:|^epochs:|^lrate:"
echo "----------------------------------------"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣ 方法分布统计"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "LoRA 方法配置:"
echo "  • SeqLoRA:   $(ls seqlora_*.yaml 2>/dev/null | wc -l) 个"
echo "  • IncLoRA:   $(ls inclora_*.yaml 2>/dev/null | wc -l) 个"
echo "  • InfLoRA:   $(ls inflora_*.yaml 2>/dev/null | wc -l) 个"
echo "  • OLoRA:     $(ls olora_*.yaml 2>/dev/null | wc -l) 个"

echo ""
echo "Fine-tuning & Linear Probe 配置:"
echo "  • FT:        $(ls FT_*.yaml 2>/dev/null | wc -l) 个"
echo "  • LP:        $(ls LP_*.yaml 2>/dev/null | wc -l) 个"

echo ""
echo "=============================================================================="
echo "验证完成!"
echo "=============================================================================="
echo ""

# 最终判定
ALL_PASS=true

for param in "init_epoch: 10" "init_lr: 0.005" "epochs: 10" "lrate: 0.005"; do
    COUNT=$(grep -c "^${param}$" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
    if [ $COUNT -ne $TOTAL_FILES ]; then
        ALL_PASS=false
        break
    fi
done

PREFIX_COUNT=$(grep -c "ep10_lr005" *.yaml 2>/dev/null | awk -F: '{s+=$2}END{print s}')
if [ $PREFIX_COUNT -ne $TOTAL_FILES ] || [ $DUPLICATE_COUNT -ne 0 ]; then
    ALL_PASS=false
fi

if [ "$ALL_PASS" = true ]; then
    echo "🎉 ✅ 所有配置验证通过! 可以安全用于实验。"
    exit 0
else
    echo "⚠️  ❌ 部分配置未通过验证，请检查上述报告。"
    exit 1
fi
