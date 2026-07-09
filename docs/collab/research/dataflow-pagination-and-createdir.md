# 任务 #44 延伸调研：CPFS 分页 & vePFS 自动建目录

调研人：researcher｜日期：2026-07-09｜只读取证，带出处。

---

## P0.5 — 阿里 CPFS/NAS `DescribeDataFlowTasks` 是否也要分页？

### 结论（先说答案）
**阿里 NAS 侧不会复现火山 vePFS 那个「不传分页→列表空」的 bug，CPFS 的 `query_task` 无需强制加分页。** 可安心。
理由：阿里用的是 **NextToken/MaxResults** 分页，`MaxResults` **有文档默认值 20**，不传分页返回的是「首页最多 20 条」而**不是空列表**；而火山 vePFS 是不传 `page_number/page_size` 时返 `total_count>0` 但列表空（真机已证）——两家分页机制不同，阿里侧无此坑。

### 依据

【文档】阿里 NAS `DescribeDataFlowTasks`（product NAS, version 2017-06-26）请求参数：

| 参数 | 必填 | 默认 |
|------|------|------|
| FileSystemId | 是 | — |
| Filters.N.Key / .Value | 否 | — |
| NextToken | 否 | —（结果被截断时才用） |
| MaxResults | 否 | **20**（取值范围 10~100） |
| WithReports | 否 | True |

- 分页机制是 **NextToken/MaxResults**，**不是** PageNumber/PageSize。两者都可选。
- `MaxResults` 原文：「每次查询结果的个数。取值范围：10~100。默认值：20。」
- `NextToken` 原文：「当请求的返回结果被截断时，您可以使用 NextToken 再次发起请求，获取从当前截断位置之后的内容。」——即只在结果被截断时才需要，首页无需传。

出处：
- https://help.aliyun.com/zh/cpfs/bmcpfs/developer-reference/api-nas-2017-06-26-describedataflowtasks-bmcpfs （CPFS 智算版 bmcpfs）
- https://help.aliyun.com/zh/cpfs/api-describedataflowtasks （CPFS）

### 对照本仓代码
`core/cpfs_dataflow/engine_nas.py::query_task`（339-379 行）当前发的是：
```
DescribeDataFlowTasks {RegionId, FileSystemId, Filters=[{Key:TaskIds, Value:task_id}], WithReports:False}
```
**没传 MaxResults/NextToken**。因为按单个 `TaskId` 过滤，命中结果至多 1 条，远在默认 `MaxResults=20` 的首页内 → 一定能取到该任务 → 不会像 vePFS 那样空列表卡 RUNNING。**不需要为「查得到任务」而加分页。**

### 建议（非必须）
- 【推断/防御性】阿里侧属文档明确 default=20，落在安全区，故**不强制修**。若 dev 想与 vePFS 修复对齐做防御性加固，可给 `query_task` 显式补 `MaxResults: 100`（成本几乎为零、消除任何「默认值将来变动」隐患），但**不是当前 bug**。
- 【可选真机核实】如需 100% 坐实，可在 prod 容器只读跑一次真实 CPFS 任务的 `DescribeDataFlowTasks`（带/不带 MaxResults 各一次）对比返回条数——非破坏性只读查询。目前无证��表明有必要。

---

## P3 — 火山 vePFS `CreateDataFlowTask` 是否支持自动新建目标目录？

### 结论（先说答案）
**支持，而且是「默认自动建」——vePFS 不需要、也没有类似阿里 CPFS 智算版 `CreateDirIfNotExist` 的开关。**
官方文档明说：创建手动任务时「**若目录不存在则会直接创建该目录**」。所以：
- **预热（Import, TOS→vePFS）**：vePFS 目标 `SubPath` 不存在 → **自动创建**，不报错。
- **沉降（Export, vePFS→TOS）**：TOS 目标前缀是对象存储虚拟目录，本就无需预建（写对象即成路径）。

### 依据

【文档】火山「文件存储 vePFS」多篇官方文档一致表述：
> 「请确认手动输入目录的准确性，**若目录不存在则会直接创建该目录**。」

出处：
- 创建手动任务：https://www.volcengine.com/docs/6645/1536434
- 数据流动概述：https://www.volcengine.com/docs/6645/1536432

【实测/SDK 反查】vePFS SDK `CreateDataFlowTaskRequest` 全部入参（prod 容器 `volcenginesdkvepfs` swagger_types/attribute_map 反查）：

| snake_case | API 字段名 (attribute_map) |
|------------|---------------------------|
| task_action | TaskAction |
| file_system_id | FileSystemId |
| data_storage | DataStorage |
| data_storage_path | DataStoragePath |
| data_type | DataType |
| sub_path | SubPath |
| fileset_id | FilesetId |
| same_name_file_policy | SameNameFilePolicy |
| delete_policy | DeletePolicy |
| export_symlink_policy | ExportSymlinkPolicy |
| enable_tls_log | EnableTlsLog |
| entry_list_file_info | EntryListFileInfo |

**没有** `CreateDirIfNotExist`（或任何等价字段）——印证「自动建目录」是缺省行为、无需开关。对比阿里 CPFS 智算版 `submit_task` 里显式下发的 `CreateDirIfNotExist=True`（engine_nas.py:324-325）在 vePFS 侧**无对应参数、也不需要**。

### 前置条件（别和「目标目录自动建」混淆）
【文档】创建任务的前提（出处同上 6645/1536434）：
> 「已在文件系统中创建数据流动的目录，可为 Fileset 或子目录，若为 Fileset，请先创建 Fileset。」

即：**数据流动的「绑定作用域目录/Fileset」需要预先在文件系统里存在**（这是 `SubPath` 的上层作用域 / `FilesetId`）；但在该作用域内、任务实际写入的**具体子目录**若不存在会被自动创建。约束还包括：配置的 Fileset/目录/前缀不能与其它「创建中/运行中/取消中」任务嵌套；需已建 TOS 桶并配好数据流动带宽。

### 对照本仓代码
`core/vepfs_dataflow/engine_vepfs.py::submit_task`（114-148 行）现有 kwargs：`file_system_id / task_action / data_type / data_storage / data_storage_path / same_name_file_policy (+ 可选 fileset_id / sub_path)`。
- **无需新增任何「建目录」参数**——vePFS 缺省即建。
- dev 若想显式支持「自动建目标目录」，答案是：**已经是默认行为，不用做任何事**；无开关可加、也没必要加。
- 【提醒】文档同时说明「重新执行数据流动任务，原数据会被覆盖」（重复跑同任务的语义），与 `same_name_file_policy` 的 Skip/KeepLatest/OverWrite 是两个层面，注意别混。

---

## 可信度标注汇总
- P0.5 分页机制/默认值 = 【文档】（阿里官方 API 参考，NextToken/MaxResults，default 20）。
- P0.5「CPFS 无 vePFS 空列表 bug」= 【文档】推得（不同分页机制 + 单 TaskId 过滤在首页内）；如需坐实可【真机只读】补一次对比查询。
- P3「vePFS 目录不存在则自动创建」= 【文档】（火山官方两处明说）。
- P3「SDK 无 CreateDirIfNotExist 字段」= 【实测/SDK 反查】（prod 容器 volcenginesdkvepfs）。
