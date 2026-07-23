# 两个 Bitable 巡检真机侦察报告

侦察日期 2026-07-10。只读探针（SSH bot-server/bot-new，docker exec、redis-cli GET/--scan、docker logs、python 打印 settings、读源码）。未做任何写/重启/改配置。

## TL;DR（结论先行）

1. **容量巡检没有空转，正在真跑真写。**【实测】旧机昨天(2026-07-09)完整跑通：OSS 86.7 TB(11 厂家) + TOS 46.1 TB(8 厂家)，合计 132.8 TB；推了合并卡；Bitable upsert 完成（快照1 + 厂家19 + 批次83）。Redis 有 2 个新鲜 `capacity:snapshot:*`（TTL 剩 ~29.4 天 = ~14h 前写入）。
2. **「TARGETS 经 bool() 判空」是误判。**【实测】settings 上根本**没有** `CAPACITY_MONITOR_TARGETS` 这个属性——真属性叫 `CAPACITY_MONITOR_TARGETS_RAW`。`getattr(s,'CAPACITY_MONITOR_TARGETS')` 会抛 `AttributeError`（我实测抛了），若用 `getattr(...,None)` 兜底则得 None→`bool(None)=False`→看着「空」。真值 `_RAW` 非空（157 字节），解析出 2 个 target。
3. **两机 `.env` 都没设 `CAPACITY_MONITOR_TARGETS`。**【实测】走 settings.py 里的**默认值**：`oss://wuji-bucket-hangzhou/third-party-data/` + `tos://wuji-egocentric-data/third-party-data/`。这俩不是占位假桶——是真桶，扫出了 132.8 TB。
4. **用户问的那张表 `EaN7bkOvsa7uT9srw4Kcq41vnHc / tbl1ydz7e2shh08Q` = 数据集大盘（`DATASET_DASHBOARD_*`），不是容量巡检的表。**【实测】token/table_id 逐字对上。它由 `core/dataset_dashboard.py` 维护，还活着但目前只读不写（见下）。
5. **数据集大盘在跑、能读表，但当前一行没写（74 行全 skip）。**【实测】需 dry-run 才能定性是「稳态无可填」还是「探测静默失败」——列为待确认，我没跑写。
6. **迁到新机（开 SCHEDULER_ENABLED 后）两巡检会正常接管**（配置逐字相同），但有**双写同表**风险（见风险）。

## 逐项证据

### 1. CAPACITY_MONITOR_TARGETS 实况【实测】
- settings.py:267-272：`CAPACITY_MONITOR_ENABLED`（默认 false）+ `CAPACITY_MONITOR_TARGETS_RAW`（默认内嵌 2 个 target 的 JSON 串）。解析在 `capacity_monitor._targets()`：`json.loads(CAPACITY_MONITOR_TARGETS_RAW)`，非 list 或异常→`[]`→`_run_capacity_scan_inner` 记「[Capacity] 无巡检目标，跳过」并 no-op。
- 旧机实测：`CAPACITY_MONITOR_TARGETS_RAW` type=str len=157 bool=True；`json.loads` 出 list count=2：
  - `vendor=oss bucket=wuji-bucket-hangzhou prefix=third-party-data/ region=None`
  - `vendor=tos bucket=wuji-egocentric-data prefix=third-party-data/ region=None`
- 两机 `.env` 都**不含** `CAPACITY_MONITOR_TARGETS` 键（grep 无命中）→ 都吃默认值。开关：两机 `CAPACITY_MONITOR_ENABLED=true`、`CAPACITY_BITABLE_ENABLED=true`、`DATASET_DASHBOARD_ENABLED=true`。
- **空值时的行为**：`_targets()` 返 `[]` → `run_capacity_scan` 直接 no-op（不是崩）。但当前非空，不触发这条。

### 2. 容量巡检真跑真写的痕迹【实测】
旧机 `docker logs aiops-bot` 昨天序列：
```
2026-07-09 13:37:28 [Scheduler] 启动：... + 容量巡检 + ... + 数据集大盘维护 + 数据流动/迁移对账
2026-07-09 16:46:21 [Capacity] oss wuji-bucket-hangzhou/third-party-data/ 盘点完成（11 个厂家，合计 86.7 TB）
2026-07-09 17:14:25 [Capacity] tos wuji-egocentric-data/third-party-data/ 盘点完成（8 个厂家，合计 46.1 TB）
2026-07-09 17:14:26 [Capacity] 已推送合并卡片（2 个 target，合计 132.8 TB）
2026-07-09 17:14:32 [Bitable] 厂家总量表展示顺序与云厂商分组不符，本次整表重建排序
2026-07-09 17:16:54 [Bitable] upsert 完成：快照1 + 厂家19 + 批次83
2026-07-10 02:25:04 [Scheduler] 启动：...（今日重部署/重启）
```
- 无任何「无巡检目标 / 取数失败 / 无子目录 / 无可推送 / 解析失败」告警。
- Redis 键（`capacity:*`）：`snapshot:oss:...`、`snapshot:tos:...`（各 1，共 2）、`batch:oss:...`、`batch:tos:...`、`flatscan:oss:...`、`flatscan:tos:...`。snapshot TTL 剩 2538973/2540657 秒 ≈ 29.4 天（30 天 TTL）→ ~14h 前写入，新鲜，与昨日 17:14 那次扫描吻合。
- **结论：容量巡检当前 = 正常工作，非空转。**巡检间隔 `CAPACITY_MONITOR_INTERVAL_HOURS`（默认 6h）。

### 3. 数据集大盘【实测 + 待确认】
- `core/dataset_dashboard.py`：`run_once()` 遍历 Bitable 现有行，按 `uri` 列扫对象存储回填**脚本负责列**（状态/云/厂商|来源/时长/数据集类型），绝不碰人工分析列。表定位来自 `settings.DATASET_DASHBOARD_APP_TOKEN` + `DATASET_DASHBOARD_TABLE_ID`。凭证：OSS 走全局 AK（open_id=""），TOS 静态 AK，`_tenant_token` 复用 `FEISHU_APP_ID/SECRET`。
- 旧机实测 token：`DASH_APP=EaN7bkOvsa7uT9srw4Kcq41vnHc`、`DASH_TABLE=tbl1ydz7e2shh08Q`。
- 日志仅 1 次运行：`2026-07-09 16:02:18 [DatasetDash] 维护：共74 在库0 已消失0 补列0 写入0 跳过74`。间隔 `DATASET_DASHBOARD_INTERVAL_HOURS=24`（默认 24h），故只跑一次。
- **能读表**（74 行读到），**但一行没写**（写入0、跳过74、在库0、已消失0）。意味着每行 `compute_updates` 都返回 `{}`。可能原因（未证实，**待确认**）：
  - (a) 稳态——74 行的 云/厂商/时长/类型 都已填好、且都是 `cpfs://`/`vepfs://` 或无法识别 scheme 的行（那类行按设计跳过存在性判断、不写状态）→ 正常无可填；或
  - (b) OSS/TOS 存在性探测对这些行返回 None（凭证/桶不可达）→ 状态从不刷新。
  - 注：容量巡检用同一条 open_id="" 全局 AK 路径扫 OSS 成功（86.7TB），故全局 OSS 凭证本身是通的；(b) 若成立更可能是这些行指向别的桶/或 uri 列名/内容问题。**定性需一次 `run_once(dry_run=True)` 看 preview——属写探测边界，我没跑，交 dev 定夺。**
- **结论：数据集大盘在维护（读表、状态机在跑），但当前净写入为 0。是否健康待一次 dry-run 确认。**

### 4. 用户那张表归属【实测，锤定】
- 用户 URL：`base/EaN7bkOvsa7uT9srw4Kcq41vnHc?table=tbl1ydz7e2shh08Q`。
- `DATASET_DASHBOARD_APP_TOKEN=EaN7bkOvsa7uT9srw4Kcq41vnHc`、`DATASET_DASHBOARD_TABLE_ID=tbl1ydz7e2shh08Q` → **逐字命中**。
- 容量巡检的表是另一套：`CAPACITY_BITABLE_APP_TOKEN=MUfFbmOpSa8vaAsVlDqc5xzsnzp`（三表 `tbl6FQGhWmaC1XOJ`/`tbl5w5A0uzEof6e8`/`tbloSSUxKZGRTBTT`），与用户 URL 不同 base。
- **结论：用户问的那张表 = 数据集大盘，由 `core/dataset_dashboard.py` 写（脚本列），人工分析列由用户自己维护。它还活着（每 24h 被遍历刷新）。**

### 5. 迁新机后接管【实测配置 + 推测行为】
- 新机 `.env`/settings 与旧机**逐字相同**：TARGETS 2 个同桶、三开关 true、CAP/DASH 的 app token/table_id 相同。当前 `SCHEDULER_ENABLED=false` → 调度器未起 → 两巡检都不跑。
- 【推测】把新机 `SCHEDULER_ENABLED=true` 后，两巡检会正常接管、写入**同一批** Bitable（token 相同）。首扫因新机 Redis 无 `capacity:snapshot/batch/flatscan` 缓存 → 全量重扫（egoverse 平铺家 ~35min 级、批次缓存/flatscan 从零重建），且增量列显示「首次」——性能/展示影响，非正确性问题。

## 风险 / 建议（交 planner/dev）
- **[风险·双写] 两机同时 live 会双写同一 Bitable。**capacity 的单飞锁 `capacity:scan:lock` 只在**本机 Redis**（redis-agents 各机一套），跨机不互斥。若新旧机调度器同时开→同一时刻可能两个 upsert 打同一张厂家/批次表（`_list_records`→`_index`→原地 upsert + 删未出现旧行），互相把对方刚写的行当「本次未出现」删掉/重建、抖动。**迁移应确保同一时刻只有一台开 SCHEDULER_ENABLED。**
- **[整洁·非阻塞] 目标靠 settings.py 默认值而非 .env 显式配置。**当前有效但脆——谁改了默认值或 .env 显式设空串就会静默改变/停掉巡检。建议把真实 targets 写进两机 `.env` 的 `CAPACITY_MONITOR_TARGETS`，别依赖代码默认。
- **[待确认] 数据集大盘净写入 0**：跑一次 `python -m ...` 或容器内 `dataset_dashboard.run_once(dry_run=True)` 看 preview，确认是稳态还是探测失败。（写探测，researcher 未执行。）
- **[名词纠偏] 排障时别再找 `settings.CAPACITY_MONITOR_TARGETS`**（不存在，会 AttributeError/被兜底判空误导）——真属性是 `CAPACITY_MONITOR_TARGETS_RAW`，解析入口 `capacity_monitor._targets()`。

## 相关源码/配置点
- `core/capacity_monitor.py`：`_targets()`(268-274 解析)、`run_capacity_scan`/`_run_capacity_scan_inner`(296-385 主流程 + 空 targets no-op + 写 Bitable 分支 377-384)。
- `core/capacity_bitable.py`：`write_scan`(179) upsert 三表；`_is_configured`(60) 校验 4 个 token。
- `core/dataset_dashboard.py`：`run_once`(226)、`compute_updates`(182)、`_oss_probe`/`_tos_probe`(140/157)。
- `config/settings.py`:267-291（两组配置键 + 默认值）。
