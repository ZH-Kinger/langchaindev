# Changelog

所有版本变更记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased]

### Added
- 跨云数据迁移（`core/transfer`，一期 TOS→OSS）：用户给路径自动解析方向/推导目的，阿里在线迁移服务（hcs_mgw）建址→建任务→启动→轮询；飞书确认表单卡（同名策略单选）+ 受理/进度/终态卡、Agent 工具 `manage_transfer`、CLI（`python -m core.transfer.cli plan|apply|status`）三入口共享核心；>1TB 走管理员审批。方向判断已含双向，二期补火山引擎 OSS→TOS、三期接 CPFS/VePFS 沉降段
- OSS 权限同步（`core/oss_perm`）：飞书多维表格 → 每人最小权限 RAM 策略，桶级/目录级两档粒度，飞书表单卡选择性下发（粒度单选 + 成员多选默认全选 + 一键确认）
- 集群算力效率（MFU）日报工具 `cluster_mfu`：多区域、交互式区域切换卡片、24h 快照、按钮回调秒回
- 容量巡检结果写入飞书多维表格（`capacity_bitable`，巡检快照→厂家总量→批次明细 三表关联）
- 每日早报：实例汇总 + 集群 MFU 双卡（北京 9:00）
- `deploy.ps1` 一键部署脚本（bind-mount 免 rebuild）
- GitHub Actions：PR Check / Release / Sync-to-Public 三条流水线
- Jira 工作流查询工具（jira_workflow_tool）
- GitHub 工作流查询工具（github_workflow_tool）

### Changed
- OSS 权限对账卡移除「孤儿策略（建议回收）」展示节（命令行 `--audit` 仍输出）

---

## [1.0.0] - 2026-04-23

### Added
- 飞书 Bot Webhook 服务（/feishu/event）
- GPU 资源申请卡片流程（Jira 工单 + DSW 实例自动创建）
- Prometheus 监控工具、GPU 训练建议工具
- 集群健康看板工具
- LangChain Agent 多工具路由