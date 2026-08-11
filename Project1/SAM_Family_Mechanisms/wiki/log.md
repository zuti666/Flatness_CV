# 实验日志

本页只登记实际发生的实现、运行和验收。理论预期放在方法/指标页，尚未运行的数值不得填入结果栏。

## 状态看板

| 编号 | 实验 | 实现状态 | 运行状态 | 结果状态 |
| --- | --- | --- | --- | --- |
| E001 | 20 维精确 Hessian 二次算子实验 | 本次实现范围 | 待按产物验收 | 尚无已登记数值结果 |
| E002 | Two Moons 非凸轨迹实验 | planned，尚未实现 | 未运行 | 无结果 |
| E003 | 小型 FashionMNIST 端点验证 | planned，尚未实现 | 未运行 | 无结果 |

## 2026-08-11｜Wiki 初始化

- 建立三级证据链：E001 算子层、E002 轨迹层、E003 端点层；
- 将本次范围锁定为 E001，E002/E003 仅保留 planned 设计；
- 固定统一 $g/c/d$ 对象和 GAM 三对象语义；
- 固定 fixed-step、fixed-budget 与 matched-SAM 半径协议；
- 固定精确 $Q_0/Q_1$ oracle、CLI 入口和输出 schema；
- 尚未在此日志登记任何运行得到的具体数值。

## E001 首次运行待办

- [ ] 使用 `run_quadratic.py --config configs/quadratic.yaml` 运行标准配置；
- [ ] 确认必需的 JSON、NPZ、CSV 和四张 PNG 完整；
- [ ] 确认 JSON 无 NaN/Infinity；
- [ ] 验证 SAM 与 GAM 的二次模型解析恒等式；
- [ ] 验证两种路径协议、$k=1$ 退化和 matched-SAM 半径；
- [ ] 验证 $Q_0/Q_1$ oracle 上界与嵌套 $R^2$ 单调性；
- [ ] 用相同命令复跑，比较核心 CSV；
- [ ] 登记 run ID、命令、代码版本、产物路径和观测结论边界。

## 运行记录模板

复制以下小节，为每次真实运行建立唯一记录。

```markdown
### E001 / RUN-YYYYMMDD-HHMM-短标识

- 状态：running / passed / failed
- 代码版本：
- 完整命令：
- 配置文件：
- 输出目录：
- manifest schema：
- dtype：
- 解析后 rho_scales / primary_rho_scale：
- inner_steps / path_protocols：
- 计算预算核对：
- 产物完整性：
- 解析恒等式验收：
- 路径与 oracle 验收：
- 确定性复跑：
- 观测结果（只填真实数值）：
- 可以支持的结论：
- 不能支持的结论：
- 异常与后续动作：
```

## E002/E003 启动规则

E002 只有在 E001 完成实现、运行和解析验收后才从 `planned` 改为 `in progress`。E003 只有在 E002 形成可复核的稳定机制假设后才启动。状态变化必须附日期、责任范围和可追溯产物，不能因 Wiki 已描述设计就标记完成。
