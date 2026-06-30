# 阿里云 OSS 与 CPFS/CPFS 智算版数据加载和沉降 OpenAPI

本文档整理的是阿里云底层 OpenAPI，不是本项目 Agent API。适用于把 OSS 数据加载到 CPFS/CPFS 智算版，以及把 CPFS/CPFS 智算版数据沉降回 OSS。

> 说明：用户口径里的 CEPFS，在阿里云 NAS OpenAPI 中对应 CPFS 通用版或 CPFS 智算版。接口产品为 `NAS`，版本为 `2017-06-26`。

## 1. 接口归属

- 产品：`NAS`
- 版本：`2017-06-26`
- OpenAPI 风格：RPC
- Endpoint：按地域访问 NAS OpenAPI，例如 `nas.cn-hangzhou.aliyuncs.com`
- 鉴权：阿里云 AK/SK 或 STS，建议通过 Alibaba Cloud CLI/SDK 处理签名
- 主要 RAM 权限：
  - `nas:CreateDataFlow`
  - `nas:DescribeDataFlows`
  - `nas:CreateDataFlowTask`
  - `nas:DescribeDataFlowTasks`
  - `nas:CancelDataFlowTask`
  - `nas:DeleteDataFlow`

## 2. 整体调用链路

1. 创建或复用 DataFlow。
   - DataFlow 是 CPFS 文件系统路径或 Fileset 与 OSS Bucket/Prefix 之间的绑定关系。
2. 创建 DataFlowTask。
   - `TaskAction=Import`：OSS -> CPFS，数据加载。
   - `TaskAction=Export`：CPFS -> OSS，数据沉降。
3. 轮询任务状态。
   - 使用 `DescribeDataFlowTasks` 获取 `Pending`、`Executing`、`Completed`、`Failed`、`Canceled` 等状态。
4. 必要时取消任务。
   - 使用 `CancelDataFlowTask`。
5. 绑定关系不再使用时删除 DataFlow。
   - 使用 `DeleteDataFlow`。

## 3. 前置条件和限制

- OSS Bucket 和 CPFS 文件系统必须在同一地域，不支持跨地域 DataFlow。
- 源端存储当前仅支持 OSS。
- CPFS 通用版支持 DataFlow 的版本要求为 2.2.0 及以上。
- CPFS 智算版支持 DataFlow 的版本要求为 2.4.0 及以上。
- CPFS 智算版跨账号访问 OSS 需要 2.6.0 及以上，并在 `SourceStorage` 中带上账号 ID。
- 文件系统必须处于 `Running` 状态才能创建 DataFlow。
- 单个 CPFS/CPFS 智算版文件系统最多创建 10 个 DataFlow。
- OSS Bucket 需要有标签：`cpfs-dataflow=true`。使用期间不要删除或修改该标签。
- 需要服务关联角色：
  - `AliyunServiceRoleForNasOssDataflow`
  - `AliyunServiceRoleForNasEventNotification`
- 如果多个 DataFlow 使用同一个 OSS Bucket，或多个 CPFS 向同一个 OSS 源端导出，建议开启 OSS 版本控制，避免对象冲突。
- CPFS 通用版 DataFlow 目标是 Fileset：
  - Fileset 必须提前存在。
  - 一个 Fileset 只能绑定一个 DataFlow。
  - Fileset 文件数限制为 1,000,000。
  - 如果 Fileset 中已有数据，创建 DataFlow 会清空原数据，并替换为从 OSS 同步的数据。
- 自动刷新依赖 EventBridge，可能产生 EventBridge 费用。由 CPFS 创建的 EventBridge 总线和规则不应手动修改或删除。

## 4. CreateDataFlow

创建 CPFS/CPFS 智算版文件系统与 OSS 的数据流动绑定关系。

### 4.1 关键参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `RegionId` | 是 | 地域 ID，例如 `cn-hangzhou`。 |
| `FileSystemId` | 是 | CPFS 文件系统 ID。CPFS 通用版通常为 `cpfs-` 开头，CPFS 智算版通常为 `bmcpfs-` 开头。 |
| `SourceStorage` | 是 | OSS 源端，格式为 `oss://[<account-id>:]<bucket>`。同账号可不写 account-id，跨账号场景需要写。 |
| `SourceSecurityType` | 否 | 访问源端的安全类型，可选 `SSL`。 |
| `Description` | 否 | DataFlow 描述。 |
| `DryRun` | 否 | 是否只预检，不真正创建。 |
| `ClientToken` | 否 | 幂等 Token。 |

### 4.2 CPFS 通用版参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `FsetId` | 是 | Fileset ID。 |
| `Throughput` | 是 | DataFlow 吞吐，常见可选值 `600`、`1200`、`1500`，部分规格还支持 `2000`。需要小于文件系统 IO 带宽。 |
| `AutoRefreshPolicy` | 否 | 自动刷新策略，`None` 或 `ImportChanged`。 |
| `AutoRefreshInterval` | 否 | 自动刷新间隔。 |
| `AutoRefreshs.N.RefreshPath` | 否 | 自动刷新路径。 |

### 4.3 CPFS 智算版参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `SourceStoragePath` | 是 | OSS Bucket 内部路径，必须以 `/` 开始和结束，长度 1-1023。 |
| `FileSystemPath` | 是 | CPFS 文件系统内路径，必须以 `/` 开始和结束，目录需要存在，长度 1-1023。 |

### 4.4 返回值

| 字段 | 说明 |
| --- | --- |
| `RequestId` | 请求 ID。 |
| `DataFlowId` | 创建出的 DataFlow ID，后续任务接口需要使用。 |

### 4.5 CLI 示例

CPFS 通用版：

```bash
aliyun nas CreateDataFlow \
  --RegionId cn-hangzhou \
  --FileSystemId cpfs-xxxx \
  --FsetId fset-xxxx \
  --SourceStorage oss://your-bucket \
  --Throughput 600 \
  --DryRun true
```

CPFS 智算版：

```bash
aliyun nas CreateDataFlow \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --SourceStorage oss://your-bucket \
  --SourceStoragePath /dataset/ \
  --FileSystemPath /dataset/ \
  --DryRun true
```

## 5. DescribeDataFlows

查询 DataFlow 绑定关系和运行状态。

### 5.1 关键参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `FileSystemId` | 是 | CPFS 文件系统 ID。 |
| `Filters.N.Key` | 否 | 过滤字段，常用 `DataFlowIds`、`FsetIds`、`FileSystemPath`、`SourceStorage`、`Status`、`SourceStoragePath`。 |
| `Filters.N.Value` | 否 | 过滤值。 |
| `NextToken` | 否 | 翻页 Token。 |
| `MaxResults` | 否 | 每页数量，10-100，默认 20。 |

### 5.2 关键返回字段

| 字段 | 说明 |
| --- | --- |
| `DataFlowId` | DataFlow ID。 |
| `Status` | 状态：`Starting`、`Running`、`Updating`、`Deleting`、`Stopping`、`Stopped`、`Misconfigured`。 |
| `ErrorMessage` | 异常原因，例如 `SourceStorageUnreachable`、`ThroughputTooLow`。 |
| `SourceStorage` | OSS 源端。 |
| `SourceStoragePath` | OSS 源端路径。 |
| `FileSystemPath` | CPFS 路径。 |
| `FsetId` | CPFS 通用版 Fileset ID。 |
| `Throughput` | DataFlow 吞吐。 |

### 5.3 CLI 示例

```bash
aliyun nas DescribeDataFlows \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --Filters.1.Key DataFlowIds \
  --Filters.1.Value df-xxxx
```

## 6. CreateDataFlowTask

创建一次具体的数据加载、沉降或清理任务。该接口是异步接口，返回 `TaskId` 后需要轮询状态。

### 6.1 关键参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `RegionId` | 是 | 地域 ID。 |
| `FileSystemId` | 是 | CPFS 文件系统 ID。 |
| `DataFlowId` | 是 | DataFlow ID。 |
| `TaskAction` | 是 | 任务动作。数据加载用 `Import`，数据沉降用 `Export`。 |
| `DataType` | 是 | 数据类型。`Metadata` 仅导入元数据，`Data` 导入或导出数据块，`MetaAndData` 同时处理元数据和数据。CPFS 智算版仅支持 `MetaAndData`。 |
| `Directory` | 三选一 | 处理单个目录。必须以 `/` 开始和结束。 |
| `EntryList` | 三选一 | JSON 字符串列表。Import 时元素是 OSS 对象名，Export 时元素是 CPFS 文件路径。总长度小于 64 KB。 |
| `TransferFileListPath` | 三选一 | CSV 清单所在 OSS 目录。仅 CPFS 智算版支持，可用于 Import/Export。 |
| `DstDirectory` | 否 | 目标映射目录。Import 时相对 `FileSystemPath`，Export 时相对 `SourceStoragePath`，必须以 `/` 开始和结束，不能包含 `/../`。 |
| `ConflictPolicy` | CPFS 智算版必填 | 同名冲突策略。 |
| `CreateDirIfNotExist` | 否 | Import 时目标目录不存在是否创建。仅 CPFS 智算版 2.6.0 及以上支持，默认 `false`。 |
| `Includes` | 否 | 目录过滤条件，仅指定 `Directory` 时可用，仅 CPFS 智算版支持。 |
| `DryRun` | 否 | 是否只预检。 |
| `ClientToken` | 否 | 幂等 Token。 |

### 6.2 TaskAction

| 值 | 方向 | 说明 |
| --- | --- | --- |
| `Import` | OSS -> CPFS | 数据加载。 |
| `Export` | CPFS -> OSS | 数据沉降。 |
| `StreamImport` | OSS -> CPFS | 流式加载，仅 CPFS 智算版 2.6.0 及以上支持。 |
| `StreamExport` | CPFS -> OSS | 流式沉降，仅 CPFS 智算版 2.6.0 及以上支持。 |
| `Evict` | CPFS 本地清理 | 清理已缓存数据。 |
| `Inventory` | 清单 | 生成或处理清单类任务。 |

CPFS 智算版仅支持 `Import`、`Export`、`StreamImport`、`StreamExport`。

### 6.3 ConflictPolicy

| 值 | 行为 |
| --- | --- |
| `SKIP_THE_FILE` | 目标存在同名文件时跳过，不覆盖。 |
| `KEEP_LATEST` | 比较更新时间，保留最新版本。 |
| `OVERWRITE_EXISTING` | 强制覆盖目标已有文件。 |

这三个策略不比较文件内容是否一致。`KEEP_LATEST` 看更新时间，`SKIP_THE_FILE` 看是否同名存在，`OVERWRITE_EXISTING` 直接覆盖。

### 6.4 TransferFileListPath 清单格式

`TransferFileListPath` 指向 OSS 上一个目录，目录中放 CSV 文件。CSV 需要包含：

| 列名 | 说明 |
| --- | --- |
| `Name` | 文件或目录路径。 |
| `Type` | `dir` 或 `file`。 |

`TransferFileListPath` 与 `Directory`、`EntryList` 互斥。

### 6.5 返回值

| 字段 | 说明 |
| --- | --- |
| `RequestId` | 请求 ID。 |
| `TaskId` | 任务 ID。 |

### 6.6 CLI 示例

OSS 加载到 CPFS 智算版：

```bash
aliyun nas CreateDataFlowTask \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --DataFlowId df-xxxx \
  --TaskAction Import \
  --DataType MetaAndData \
  --Directory /dataset/ \
  --DstDirectory /dataset/ \
  --ConflictPolicy SKIP_THE_FILE \
  --CreateDirIfNotExist true
```

CPFS 智算版沉降到 OSS：

```bash
aliyun nas CreateDataFlowTask \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --DataFlowId df-xxxx \
  --TaskAction Export \
  --DataType MetaAndData \
  --Directory /outputs/ \
  --DstDirectory /exports/ \
  --ConflictPolicy KEEP_LATEST
```

指定文件列表加载：

```bash
aliyun nas CreateDataFlowTask \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --DataFlowId df-xxxx \
  --TaskAction Import \
  --DataType MetaAndData \
  --EntryList '[\"/dataset/a.csv\",\"/dataset/b.csv\"]' \
  --DstDirectory /dataset/ \
  --ConflictPolicy OVERWRITE_EXISTING
```

## 7. DescribeDataFlowTasks

查询 DataFlowTask 状态、进度和报告。

### 7.1 关键参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `FileSystemId` | 是 | CPFS 文件系统 ID。 |
| `Filters.N.Key` | 否 | 过滤字段，常用 `DataFlowIds`、`TaskIds`、`TaskActions`、`DataTypes`、`Status`、`CreateTimeBegin`、`CreateTimeEnd`、`StartTimeBegin`、`StartTimeEnd`、`EndTimeBegin`、`EndTimeEnd`。 |
| `Filters.N.Value` | 否 | 过滤值。 |
| `WithReports` | 否 | 是否返回报告信息。CPFS 智算版支持，默认 `true`；只查状态时可设为 `false`。 |
| `NextToken` | 否 | 翻页 Token。 |
| `MaxResults` | 否 | 每页数量，10-100，默认 20。 |

### 7.2 任务状态

| 状态 | 说明 |
| --- | --- |
| `Pending` | 等待执行。 |
| `Executing` | 执行中。 |
| `Completed` | 完成。 |
| `Failed` | 失败。 |
| `Canceling` | 取消中。 |
| `Canceled` | 已取消。 |

### 7.3 进度字段

| 字段 | 说明 |
| --- | --- |
| `FilesTotal` | 预计处理文件数。 |
| `FilesDone` | 已处理文件数。 |
| `ActualFiles` | 实际处理文件数。 |
| `BytesTotal` | 预计处理字节数。 |
| `BytesDone` | 已处理字节数。 |
| `ActualBytes` | 实际处理字节数。 |
| `RemainTime` | 预计剩余时间。 |
| `AverageSpeed` | 平均速度。 |

CPFS 通用版任务报告通常写入文件系统 `.dataflow_report` 目录。CPFS 智算版在 `WithReports=true` 时可返回 OSS 报告下载链接。

### 7.4 CLI 示例

```bash
aliyun nas DescribeDataFlowTasks \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --Filters.1.Key TaskIds \
  --Filters.1.Value task-xxxx \
  --WithReports false
```

## 8. CancelDataFlowTask

取消 DataFlowTask。

### 8.1 关键参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `RegionId` | 是 | 地域 ID。 |
| `FileSystemId` | 是 | CPFS 文件系统 ID。 |
| `DataFlowId` | 是 | DataFlow ID。 |
| `TaskId` | 是 | Task ID。 |
| `DryRun` | 否 | 是否只预检。 |
| `ClientToken` | 否 | 幂等 Token。 |

取消操作建议尽早发起。文档中对可取消状态存在描述差异，实践上应优先对 `Pending` 或刚进入 `Executing` 的任务取消，并继续轮询到 `Canceled` 或终态。

### 8.2 CLI 示例

```bash
aliyun nas CancelDataFlowTask \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --DataFlowId df-xxxx \
  --TaskId task-xxxx
```

## 9. DeleteDataFlow

删除 DataFlow 绑定关系。仅当 DataFlow 状态为 `Running` 或 `Stopped` 时支持删除。删除后不可恢复。

### 9.1 关键参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `RegionId` | 是 | 地域 ID。 |
| `FileSystemId` | 是 | CPFS 文件系统 ID。 |
| `DataFlowId` | 是 | DataFlow ID。 |
| `DryRun` | 否 | 是否只预检。 |
| `ClientToken` | 否 | 幂等 Token。 |

### 9.2 CLI 示例

```bash
aliyun nas DeleteDataFlow \
  --RegionId cn-hangzhou \
  --FileSystemId bmcpfs-xxxx \
  --DataFlowId df-xxxx
```

## 10. 常见迁移场景

### 10.1 OSS 数据加载到 CPFS

1. 确认 OSS Bucket 与 CPFS 同地域。
2. 确认 OSS Bucket 标签 `cpfs-dataflow=true`。
3. 确认服务关联角色已创建。
4. 调用 `CreateDataFlow` 建立绑定。
5. 等待 `DescribeDataFlows` 返回 `Status=Running`。
6. 调用 `CreateDataFlowTask`：
   - `TaskAction=Import`
   - `DataType=MetaAndData`
   - 按目录用 `Directory`，按清单用 `EntryList` 或 `TransferFileListPath`
   - 设置合适的 `ConflictPolicy`
7. 轮询 `DescribeDataFlowTasks` 到 `Completed` 或 `Failed`。

### 10.2 CPFS 数据沉降到 OSS

1. 复用已有 DataFlow，或按目标 OSS Bucket/Prefix 新建 DataFlow。
2. 调用 `CreateDataFlowTask`：
   - `TaskAction=Export`
   - `DataType=MetaAndData`
   - `Directory` 指向 CPFS 侧相对目录
   - `DstDirectory` 指向 OSS 侧目标相对目录
   - 设置 `ConflictPolicy`
3. 轮询 `DescribeDataFlowTasks`。
4. 失败时查看任务报告或错误码。

## 11. 重要行为说明

- 同名处理不做内容 Hash 比对。
  - `SKIP_THE_FILE`：只要目标已有同名文件就跳过。
  - `KEEP_LATEST`：按更新时间判断保留哪一边。
  - `OVERWRITE_EXISTING`：无条件覆盖目标文件。
- DataFlowTask 是异步任务，返回 `TaskId` 不代表数据已经完成迁移。
- 手动任务可能打断并等待自动更新任务，应避免和自动刷新高频冲突。
- 大规模数据建议拆分成多个目录或清单任务，便于失败重试和进度跟踪。
- CPFS 智算版 Import 限制：
  - OSS 中的符号链接导入后会变成普通文件。
  - OSS 开启版本控制时，仅复制最新版本。
  - 文件或目录名超过 255 字节不支持。
- CPFS 智算版 Export 限制：
  - 符号链接导出为空对象，不会导出链接目标文件。
  - 硬链接导出为普通文件。
  - Socket、Device、Pipe 导出为空对象。
  - 目录路径最大长度为 1023 字符。

## 12. 常见错误方向

| 错误方向 | 可能原因 |
| --- | --- |
| Bucket 地域不匹配 | OSS Bucket 与 CPFS 不在同一地域。 |
| Bucket 访问拒绝 | 角色权限、跨账号配置或 Bucket Policy 不正确。 |
| 依赖 Bucket 标签失败 | OSS Bucket 缺少 `cpfs-dataflow=true`。 |
| 需要开启版本控制 | 多 DataFlow 或多 CPFS 同源导出存在对象冲突风险。 |
| SourceStorage 无效 | OSS 地址格式、地域或权限不正确。 |
| Fileset 不存在 | CPFS 通用版 `FsetId` 错误或未创建。 |
| DataFlow 状态不允许创建任务 | DataFlow 不是 `Running`。 |

## 13. 官方参考

- CreateDataFlow：https://next.api.aliyun.com/api/NAS/2017-06-26/CreateDataFlow
- CreateDataFlowTask：https://next.api.aliyun.com/api/NAS/2017-06-26/CreateDataFlowTask
- DescribeDataFlows：https://next.api.aliyun.com/api/NAS/2017-06-26/DescribeDataFlows
- DescribeDataFlowTasks：https://next.api.aliyun.com/api/NAS/2017-06-26/DescribeDataFlowTasks
- CancelDataFlowTask：https://next.api.aliyun.com/api/NAS/2017-06-26/CancelDataFlowTask
- DeleteDataFlow：https://next.api.aliyun.com/api/NAS/2017-06-26/DeleteDataFlow
