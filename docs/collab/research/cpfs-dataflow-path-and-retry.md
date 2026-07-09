# CPFS/NAS 数据流动（DataFlow）两个关键语义核实 — 任务 #42 延伸

> researcher 取证。区分【文档】= 阿里官方明说（附 URL）/【实测·推断】= 代码/行为推断。
> 核对对象：`docs/aliyun_cpfs_oss_dataflow_api.md`、`core/cpfs_dataflow/engine_nas.py`、`core/cpfs_dataflow/orchestrator.py`。

---

## 课题 A：CreateDataFlowTask 的 `Directory` / `DstDirectory` 相对谁？

### A.1 官方结论（【文档】）

`Directory` 和 `DstDirectory` **都是相对路径**，基准是 **DataFlow 绑定关系里的 `FileSystemPath`（CPFS 侧）与 `SourceStoragePath`（OSS 侧）**，**不是文件系统根的绝对路径**。二者随任务方向互换基准：

| 参数 | Import（OSS→CPFS，预热） | Export（CPFS→OSS，沉降） |
|------|--------------------------|--------------------------|
| `Directory`（源目录） | 相对 **SourceStoragePath**（OSS 侧） | 相对 **FileSystemPath**（CPFS 侧） |
| `DstDirectory`（目标映射目录） | 相对 **FileSystemPath**（CPFS 侧） | 相对 **SourceStoragePath**（OSS 侧） |

官方 CreateDataFlowTask 文档逐字（智算版 bmcpfs）：
- `Directory`：「当 TaskAction 为 Export 时，该目录必须是 **FileSystemPath 内的相对路径**。」「当 TaskAction 为 Import 时，该目录必须是 **SourceStoragePath 内的相对路径**。」
- `DstDirectory`：「当 TaskAction 为 Export 时，该目录必须是 **SourceStoragePath 内的相对路径**。」「当 TaskAction 为 Import 时，该目录必须是 **FileSystemPath 内的相对路径**。」（都须以 `/` 开头结尾，不支持 `/../`，长度 1~1023。）

> 出处：CreateDataFlowTask（智算版）https://help.aliyun.com/zh/cpfs/bmcpfs/developer-reference/api-nas-2017-06-26-createdataflowtask-bmcpfs
> CreateDataFlow（智算版，FileSystemPath / SourceStoragePath 定义）https://help.aliyun.com/zh/cpfs/bmcpfs/developer-reference/api-nas-2017-06-26-createdataflow-bmcpfs
> 流式任务最佳实践（Directory/DstDirectory 在导入/导出下的相对含义）https://help.aliyun.com/zh/cpfs/bmcpfs/user-guide/data-flow-flow-task-best-practices

**结论直答 dev 的问句**：DataFlow 的 `FileSystemPath=/wuji-il`，若把 `DstDirectory=/wuji-il/sub/` 下发（Import），阿里会把它当作 **相对 `/wuji-il` 的路径** → 落在 `<fs根>/wuji-il/wuji-il/sub/`（**双套**）。正确下发应是 `DstDirectory=/sub/`（已减去 FileSystemPath）。所以「相对 fileset / FileSystemPath」成立，**必须减去 FileSystemPath**。

### A.2 我们代码现在到底减没减？（【实测·推断，基于代码阅读】）

**减了——但只在特定分支，且 Import 方向的两个字段贴反了。** 细节：

**(1) FileSystemPath 剥离是靠 `start_task` 里的 `_relative_dir` 完成的，不是靠 make_plan。**
`orchestrator.start_task`（337-345 行）**不使用** `plan.directory`/`plan.dst_directory`，而是重算：
```
task_dir = _relative_dir(job["cpfs_dir"], job["df_fs_path"])     # 减 CPFS FileSystemPath
task_dst = _relative_dir(job["oss_prefix"], job["df_oss_path"])  # 减 OSS SourceStoragePath
submit_task(..., directory=task_dir, dst_directory=task_dst)
```
`df_fs_path`/`df_oss_path` 是**解析/新建 DataFlow 时**记下的真实绑定路径（318-332 行）。所以只要走到那个分支，`_relative_dir` 就会把用户目录减去 DataFlow 的 FileSystemPath → 正确。

**(2) 当前飞书卡片 / CLI / tool 三入口都不传 `data_flow_id`**（`_guided_plan` 走 `make_plan(..., fs_id, region)` 不带 df；`plan_from_addresses` 显式 `data_flow_id=""`）→ 所以 `start_task` 的 `if not job.get("data_flow_id")` 分支**必进** → `resolve_dataflow` 跑 → `df_fs_path` 被赋成**真实 FileSystemPath** → 减法生效。
→ **就当前所有线上路径而言，FileSystemPath 已经被减掉了，dev 不需要在 make_plan 里再手动剥一层。** #42 真机看到的双层 `cpfs/` 是**挂载前缀**问题（已修），与 FileSystemPath 是两回事。

**(3) 潜在坑（【推断】，目前没人踩，但管道已铺好）**：`DataflowPlan` 有 `data_flow_id` 字段，`discovery.decode_selection` 会返回 `{data_flow_id, fs_path, ...}`。**一旦哪天把卡片选择的 `data_flow_id` 直接塞进 `make_plan`**，`start_task` 的 `if not data_flow_id` 分支会被跳过 → `df_fs_path` **永不赋值** → `_relative_dir(cpfs_dir, "/")` 退化成不减 → **正好复现 dev 担心的双套**。
→ 建议：若要接「显式 data_flow_id 直达」，必须**同时把 discovery 的 `fs_path` 一并带进来赋给 `job["df_fs_path"]`（和 `oss_prefix`/`SourceStoragePath` 对应 `df_oss_path`）**，否则减法失效。

**(4) Import 方向 Directory/DstDirectory 贴反了（【推断，需 dev 核对】，中低优先级）**：
`start_task` 无论 Import/Export 都固定 `directory=CPFS侧相对`、`dst_directory=OSS侧相对`：
- Export（沉降）：Directory=CPFS相对 ✓、DstDirectory=OSS相对 ✓ —— **对**。
- Import（预热）：按文档应 Directory=OSS相对、DstDirectory=CPFS相对，但代码给的是 Directory=CPFS相对、DstDirectory=OSS相对 —— **两字段对调了**。
- **为什么线上没炸**：wuji 场景 OSS 前缀与 CPFS 子目录互为镜像（同名子目录），两个相对路径算出来是同一串，对调后无差别；**仅当 OSS 前缀 ≠ CPFS 子目录时才会读错 OSS 源目录 / 写错 CPFS 目标目录**。
- 注：`make_plan` 里 `plan.directory`/`plan.dst_directory` 其实是**按方向正确定向**的（预热 directory=oss_prefix、dst=cpfs_dir），只是 `start_task` 没用它、另起炉灶算 `task_dir`/`task_dst` 才贴反。dev 可考虑让 Import 分支交换 directory/dst_directory 两个入参（或直接复用 plan 的定向再做 `_relative_dir`）。

### A.3 智算版（bmcpfs-*）差异（【文档】）
- 智算版 `DataType` 只能 `MetaAndData`；`ConflictPolicy` **必填**。代码 322-323 行已按 `edition()==computing` 补 ConflictPolicy、非智算版不下发——正确。
- `CreateDirIfNotExist` **仅 Import 生效**、仅智算版 2.6.0+，默认 false；代码仅在 Import+create_dir 时下发——正确。
- FileSystemPath/SourceStoragePath 语义与通用版一致（都是相对基准），无特殊「双套」规则；双套纯粹来自「传了绝对路径却被当相对路径」。

---

## 课题 B：任务部分文件失败后如何重试 / 失败在哪看？

### B.1 失败/跳过明细在哪（【文档】）
- `DescribeDataFlowTasks` 的 **`ProgressStats`** 里有 `FilesTotal`（源端扫描到文件数）/`FilesDone`（已完成，**含跳过**）/`ActualFiles`（实际流动文件数）/`BytesTotal`/`BytesDone`/`ActualBytes`。**没有专门的 `FilesFailed`/`FilesSkipped` 计数字段** —— 失败/跳过数要靠**报告文件**看。
- 顶层 `ErrorMsg`（不是 ErrorMessage/Message）= 任务级异常原因，可能是 `{ErrorKey, ErrorDetail}` 结构。**代码 378 行已用 `ErrorMsg` 优先——正确**（历史踩坑：读错字段全回退成 Failed）。
- **报告**：任务结束按情况生成三类报告——`SkippedFilesReport`（跳过）/`FailedFilesReport`（失败）/`SuccessFilesReport`（成功）。
  - **智算版（bmcpfs-）**：`DescribeDataFlowTasks` 在 `WithReports=true` 时返回 `Reports.Report[]`，每个含 `Name`（上述三种之一）+ `Path`（**OSS 下载链接**）。
  - **通用版（cpfs-）**：报告生成在 CPFS 文件系统的 **`.dataflow_report` 目录**里。
  - `ReportPath` 字段**已废弃**（旧版返回报告在 CPFS 内的保存路径）。
  - 控制台：数据流动页 → 目标流「任务管理」→「下载任务报告」；失败任务悬停「失败」气泡可看失败原因/下载失败报告。
- **注意**：代码 `query_task`（348 行）目前写死 **`WithReports=False`**，所以线上拿不到 `Reports.Report[]` 的失败清单下载链接。要给用户「失败清单」需在查询终态时补一发 `WithReports=True`。
- 关于用户口径的 `aliyun_import_report`：**官方文档未出现该确切文件名**；官方叫法是 `FailedFilesReport`/`SkippedFilesReport`/`SuccessFilesReport`（智算版走 OSS 链接、通用版落 `.dataflow_report/`）。用户看到的具体 csv 文件名建议以真机报告为准（未取证）。

> 出处：DescribeDataFlowTasks（智算版）https://help.aliyun.com/zh/cpfs/bmcpfs/developer-reference/api-nas-2017-06-26-describedataflowtasks-bmcpfs
> 管理智算版数据流动任务（报告下载/失败原因）https://help.aliyun.com/zh/cpfs/bmcpfs/user-guide/manage-cpfs-for-lingjun-data-flow-tasks

### B.2 重跑 Import 是否只补未成功、跳过已成功？（【文档】+【推断】）
- `ConflictPolicy=SKIP_THE_FILE`【文档】：「跳过同名文件」，**只看是否同名存在、不比对内容**（`KEEP_LATEST` 看更新时间、`OVERWRITE_EXISTING` 无条件覆盖）。
- 【推断】因此对**同一 `Directory` 重跑 Import + SKIP_THE_FILE**：上一轮已成功落地的同名文件会被跳过、上轮缺失/失败的会被重传 —— **实际效果 = 增量补齐**。但注意 `FilesTotal` 仍会**全量重扫**源端（不是只扫失败清单），大目录重跑扫描开销不小。官方**没有**「只重试失败文件」的自动机制/开关，`SKIP_THE_FILE` 也**没被官方明确定义为「断点续传」**（这层是推断，非文档原话）。

### B.3 官方推荐的「增量/精确重试」做法（【文档】）
1. **用失败报告 + 文件清单做精确重试**：把 `FailedFilesReport` 里的失败文件列成清单，用 **`EntryList`**（JSON 数组，Import 时元素是 OSS 对象名；总长 <64KB）或 **`TransferFileListPath`**（OSS 上的 CSV 清单目录，含 `Name`/`Type` 两列，**仅智算版**，Import/Export 皆可）只重传失败项。`EntryList`/`TransferFileListPath`/`Directory` 三者互斥。
2. **大数据集拆多目录/多任务**：官方明确建议「大规模数据建议拆分成多个目录或清单任务，便于失败重试和进度跟踪」。
3. 控制台有「**复制任务**」可重跑此前执行过的任务；无「仅失败重试」按钮。

> 出处：CreateDataFlowTask（EntryList/TransferFileListPath/Directory 三选一、清单格式）https://help.aliyun.com/zh/cpfs/bmcpfs/developer-reference/api-nas-2017-06-26-createdataflowtask-bmcpfs
> 本仓库 `docs/aliyun_cpfs_oss_dataflow_api.md` §6.1/§6.4/§11 与上述一致。

### B.4 ConflictPolicy 各枚举对「覆盖 vs 跳过失败文件」的影响（【文档】）
| 枚举 | 行为 | 对重试的含义 |
|------|------|--------------|
| `SKIP_THE_FILE` | 目标有同名即跳过（不比内容） | 重跑=跳过已落地、补缺失（增量补齐，最稳，默认推荐） |
| `KEEP_LATEST` | 比更新时间留最新 | 源比目标新则覆盖，否则跳过；适合源可能更新过的重跑 |
| `OVERWRITE_EXISTING` | 无条件覆盖 | 重跑会把已成功的也重传一遍（费流量/时间，一般不用于纯重试） |

---

## 给 dev 的直接判断支撑

1. **「是否需要再剥一层 FileSystemPath」**：语义上 **是**（Directory/DstDirectory 相对 FileSystemPath/SourceStoragePath）；但**当前代码已在 `start_task` 用 `_relative_dir(cpfs_dir, df_fs_path)` 减掉了**，且当前三入口都不传 `data_flow_id` → 减法必然生效。**不需要在 make_plan 再加剥离**。两个要盯的点：①若将来接「显式 data_flow_id 直达」，务必把 discovery 的 `fs_path` 带进 `df_fs_path`，否则减法失效复现双套；②Import 方向 `start_task` 的 Directory/DstDirectory 贴反（镜像子目录时无害，非镜像时读写错目录），建议核对/交换。
2. **失败重试指引给用户**：查询终态时补 `WithReports=True` 拿 `Reports.Report[]` 的 `FailedFilesReport` OSS 下载链接（智算版）；重试首选**同 Directory 重跑 + `ConflictPolicy=SKIP_THE_FILE`**（跳过已成功、补缺失）；要精确只重失败项则用失败报告喂 `EntryList`/`TransferFileListPath`。`query_task` 现写死 `WithReports=False`，拿不到失败清单链接。
