# Changelog

所有版本变更记录。格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [Unreleased]

### Added
- 对话**滚动摘要记忆**（`core/agent`）：逐字保留最近 20 条，更早对话由 GLM 压成滚动摘要存 `agent:chat_summary:{sid}`（上限 1200 字），下轮作前缀注入，长对话不再硬截断。压缩批量异步（约每 5 轮一次）在回复发出后收尾，不拖慢感知；失败保留旧摘要
- 数据流动/迁移**在途任务对账**（`dsw_scheduler` `dataflow-reconcile` 循环，每 2 分钟）：后台轮询线程随容器重启会死，对账线程随容器复活兜底——重启后任务完成也自动补推结果卡（跑完必通知）。在线推送与对账推送共用 Redis `SET NX dataflow:notified:{job_id}` 闸门，跨线程只推一次
- 火山 vePFS/TOS 数据流动（`core/vepfs_dataflow`）：**预热**（TOS→vePFS）/**沉降**（vePFS→TOS）。无持久 DataFlow 对象，提交任务直接带桶/前缀，方向由地址类型自动判断；与 CPFS 共用三步级联向导卡（选云→选地区→表单）。Agent 工具 `manage_vepfs_dataflow` + CLI 三入口，`vepfs:dataflow:job:{id}` 状态机
- 阿里 CPFS/NAS 数据流动（`core/cpfs_dataflow`）：NAS DataFlow **预热**（OSS→CPFS）/**沉降**（CPFS→OSS）。只查现有 DataFlow + 提交任务，按目标子目录匹配最长前缀绑定，智算版/通用版自动分支；飞书选择器从发现的 CPFS↔OSS 绑定里选。Agent 工具 `manage_cpfs_dataflow` + CLI 三入口
- 同云桶间迁移（`core/bucket_transfer`）：同账号跨 region/桶，阿里 `oss://→oss://`、火山 `tos://→tos://`；OSS 自动探测源/目的 region，跨 region 走公网。与跨云迁移独立命名空间、仅复用引擎，混合 scheme 拒绝并提示改用跨云迁移。真机验收通过
- GPU 卡分布大盘（`tools/aliyun/gpu_distribution`）：地区×卡型分布 + 每用户在算卡数 + 近 N 小时趋势，实时 HTML 页面 `/gpu/distribution`（token 门禁），15s 陈旧后台单飞刷新；飞书问「谁在用卡/卡分布」回摘要卡 + 链接
- 数据集大盘定时维护（`core/dataset_dashboard`，`dataset-dashboard` 循环）：遍历飞书「数据集大盘」多维表格现有行，按 uri 扫对象存储回填脚本负责列（状态/云/厂商/时长/数据集类型），只填数据、不改表结构、不碰人工分析列
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
- 屏蔽 DSW **到期自动关机**：新增 `DSW_IDLE_STOP_ENABLED`（默认 `false`），到期前 15 分钟警告卡 + 到期自动停止均包进开关，利用率关机交给阿里云工作空间；**GPU 空转提醒不受开关影响、始终生效**。置 `true` 恢复旧行为
- 数据流动/迁移进度卡+结果卡**优先推发起人**（`job.created_by`），发起人为空才降级配置频道；在线推送、对账、查询三处目标一致，NX 闸门仍只推一次
- 指标趋势图**按需附带**（`feishu_bot/messages`）：仅当意图含监控/集群或问题命中指标词才附 Prometheus 趋势图，其余纯文本回复，省掉每条回复的云查询+渲染+上传
- Agent executor 加**迭代/超时上限**（`max_iterations=8` / `max_execution_time=60`），防工具调用循环烧钱或挂住；飞书对话执行器 `verbose=False`
- 会话历史 Redis **7 天空闲 TTL**（`agent:chat_history` 每轮续期），不再无限堆积
- OSS 权限对账卡移除「孤儿策略（建议回收）」展示节（命令行 `--audit` 仍输出）

### Fixed
- vePFS `DescribeDataFlowTasks` **补分页**（`page_number=1`/`page_size=100`）：不传分页时火山返 `total_count>0` 但空列表 → 任务永远卡 `RUNNING`，现终态可读；火山错误 JSON 友好化（抽 `Error.Code/Message`，已知码给人话，不再甩原始 JSON）
- CPFS/vePFS 数据流动**多套一层 `cpfs`/`vepfs` 目录**：`make_plan` 的 `fs_id` 直达分支未剥挂载前缀，导致预热落到 `/mnt/data/cpfs/...`。两条解析路径统一 `_strip_mount`（新增 `VEPFS_MOUNT_PREFIX`，默认 `/vepfs`）
- 飞书卡片回调**不再阻塞 3s 死线**：查询进度 sync 只读 Redis 秒回当前卡、后台线程 refresh 后推更新卡；GPU/AK 提交的网络调用（RAM 关联、取用户名）移入后台线程，先即时回卡再异步补推
- `tools/aliyun/oss.list_objects` 改 `itertools.islice(ObjectIterator, max_keys)` 首页截断，不再 `list()` 枚举整桶 → 百万级对象桶不再挂线程/OOM
- 数据流动/迁移对账去重加固：修 auditor 复审发现的重复推卡竞态——`bucket_transfer` 每次 `_save` 刷 `updated_ts`（补齐 stale 门），终态推送先落 `notified` 标记再抢闸门，`reconcile` 修正前缀切片
- 跨云迁移 submit/confirm 幂等：`transfer:launch` NX + MGW job_name 幂等，止住连点起多线程 + 刷卡风暴；确认后原地替换为「进行中」卡、重复解析回进度卡
- vePFS `CreateDataFlowTask` 参数对齐真机：`DataStorage` 用裸桶名，`DataStoragePath`/`SubPath` 非空须首尾带斜杠

---

## [1.0.0] - 2026-04-23

### Added
- 飞书 Bot Webhook 服务（/feishu/event）
- GPU 资源申请卡片流程（Jira 工单 + DSW 实例自动创建）
- Prometheus 监控工具、GPU 训练建议工具
- 集群健康看板工具
- LangChain Agent 多工具路由