# 火山 vePFS 数据流动 SDK 调用指南（预热 Import / 沉降 Export）

面向开发者，照着能调通 vePFS↔TOS 数据流动。字段格式以本仓 `core/vepfs_dataflow/engine_vepfs.py`（cn-shanghai 真机反查验证）为准。

- **预热**：`TaskAction=Import`，`TOS → vePFS`（对象存储加载进文件系统）。
- **沉降**：`TaskAction=Export`，`vePFS → TOS`（文件系统刷回对象存储）。

产品「文件存储 vePFS」，service `vepfs`，version `2022-01-01`。

## 1. 安装 SDK

### 官方 SDK

火山引擎官方 Python SDK，PyPI 包名 **`volcengine-python-sdk`**。

- PyPI：https://pypi.org/project/volcengine-python-sdk/
- 源码仓库：https://github.com/volcengine/volcengine-python-sdk

它是**单体大包**——一次装进来所有火山服务的子包（`volcenginesdkcore` 通用 client、`volcenginesdkvepfs` vePFS、`volcenginesdkdms` 数据迁移……）。vePFS 数据流动只用其中两个：`volcenginesdkcore` + `volcenginesdkvepfs`，无需单独安装。

### 安装命令

```bash
# 装最新版
pip install volcengine-python-sdk

# 指定版本（截至本文 PyPI 最新为 5.0.40，实际以 PyPI 最新为准）
pip install volcengine-python-sdk==5.0.40
```

包体积较大，国内可加镜像加速：

```bash
pip install volcengine-python-sdk -i https://pypi.tuna.tsinghua.edu.cn/simple
# 或阿里云镜像
pip install volcengine-python-sdk -i https://mirrors.aliyun.com/pypi/simple/
```

### 装完验证

```bash
python -c "import volcenginesdkcore, volcenginesdkvepfs; print('ok')"
```

打印 `ok` 说明两个子包都在。若报 `ModuleNotFoundError`，多半是装到了别的 Python 环境，确认 `pip` 与 `python` 是同一环境。

## 2. 凭证与地域

- 静态 AK/SK，**火山无 STS**。
- `region` 必须等于 vePFS 文件系统所在区域，且 **TOS 桶须与 vePFS 同区域**。

## 3. 构造 client

```python
import volcenginesdkcore as core
import volcenginesdkvepfs as vepfs

def make_api(ak: str, sk: str, region: str):
    cfg = core.Configuration()
    cfg.ak, cfg.sk, cfg.region = ak, sk, region
    return vepfs.VEPFSApi(core.ApiClient(cfg))
```

## 4. 提交任务：预热 / 沉降

核心是 `api.create_data_flow_task(vepfs.CreateDataFlowTaskRequest(...))`。**方向只由 `task_action` 决定，源/目的字段不反转**——预热和沉降的请求字段写法完全一样，只有 `task_action` 不同。

### 参数表（`CreateDataFlowTaskRequest`，snake_case）

| 参数 | 必填 | 说明 / 取值 |
|------|------|-------------|
| `file_system_id` | 是 | vePFS 文件系统 ID |
| `task_action` | 是 | `Import`（预热 TOS→vePFS） / `Export`（沉降 vePFS→TOS） |
| `data_type` | 是 | `MetaAndData`（元数据+数据，默认） / `Metadata`（仅元数据） |
| `data_storage` | 是 | TOS **裸桶名**，不带 `tos://`（见 §4） |
| `data_storage_path` | 否 | TOS 侧前缀，非空须首尾带斜杠 `/a/b/`；空串=整桶根 |
| `sub_path` | 否* | vePFS 侧子目录，非空须首尾带斜杠 `/a/b/` |
| `fileset_id` | 否* | vePFS 侧 Fileset（智算/配额目录）。与 `sub_path` 二选一，Fileset 优先 |
| `same_name_file_policy` | 否 | `Skip`（默认，永不覆盖） / `KeepLatest` / `OverWrite` |

\* `sub_path` 与 `fileset_id` 用于定位 vePFS 侧目录，按需给其一。不给则作用于文件系统根。

返回对象取 `data_flow_task_id`。

### 预热示例（Import：TOS → vePFS）

```python
api = make_api(ak, sk, region="cn-shanghai")

req = vepfs.CreateDataFlowTaskRequest(
    file_system_id="vepfs-xxxxxxxx",
    task_action="Import",              # 预热
    data_type="MetaAndData",
    data_storage="my-tos-bucket",      # 裸桶名，别带 tos://
    data_storage_path="/train/label/", # TOS 前缀，首尾带斜杠
    sub_path="/label/",                # vePFS 目标子目录，首尾带斜杠
    same_name_file_policy="Skip",
)
resp = api.create_data_flow_task(req)
task_id = getattr(resp, "data_flow_task_id", None)
print("data_flow_task_id:", task_id)
```

### 沉降示例（Export：vePFS → TOS）

```python
req = vepfs.CreateDataFlowTaskRequest(
    file_system_id="vepfs-xxxxxxxx",
    task_action="Export",              # 沉降（其余字段与预热同写法，不反转）
    data_type="MetaAndData",
    data_storage="my-tos-bucket",
    data_storage_path="/train/label/",
    sub_path="/label/",
    same_name_file_policy="OverWrite",
)
resp = api.create_data_flow_task(req)
task_id = getattr(resp, "data_flow_task_id", None)
```

## 5. 字段格式硬规（真机反查，写错必报错）

- **`data_storage` = 裸桶名**：`my-tos-bucket`。带 `tos://` 前缀会被判 `InvalidParameter.BucketName`；多余斜杠同样非法。
- **`data_storage_path` / `sub_path` 非空必须首尾都带斜杠**：`/a/b/`。无首斜杠、仅首斜杠、仅尾斜杠都会被判 `InvalidParameter.*`。
- **`data_storage_path` 允许空串** = TOS 整桶根（合法）。
- **`same_name_file_policy`**：`Skip`（默认，永不覆盖同名） / `KeepLatest`（比最后修改时间保新） / `OverWrite`（覆盖）。
- **`data_type`**：`MetaAndData`（元数据+数据） / `Metadata`（仅元数据）。

## 6. 轮询进度

```python
def query_task(api, task_id: str, fs_id: str) -> dict:
    # 必须带分页！不传 page_number/page_size 时火山会返回 total_count>0 但
    # data_flow_tasks=[]（空列表）→ 永远拿不到任务 → status 恒空 → 死轮询。
    req = vepfs.DescribeDataFlowTasksRequest(
        file_system_id=fs_id,
        data_flow_task_ids=str(task_id),
        page_number=1,
        page_size=100,
    )
    resp = api.describe_data_flow_tasks(req)
    tasks = getattr(resp, "data_flow_tasks", None) or []
    task = next((t for t in tasks
                 if str(getattr(t, "data_flow_task_id", "")) == str(task_id)), None)
    if task is None:
        return {}
    return {
        "status":       getattr(task, "status", "") or "",
        "bytes_total":  int(getattr(task, "total_size", 0) or 0),
        "bytes_done":   int(getattr(task, "exec_size", 0) or 0),
        "files_total":  int(getattr(task, "total_count", 0) or 0),
        "files_done":   int(getattr(task, "exec_count", 0) or 0),
        "failed_count": int(getattr(task, "failed_count", 0) or 0),
    }
```

- **必须带 `page_number` / `page_size`**（真机踩过的坑：不带分页返回 `total_count>0` 但列表为空，任务永远查不到）。
- 进度字段：`status`、`total_size`、`exec_size`、`total_count`、`exec_count`、`failed_count`。

### 判终态：status 是自由 str，非受限枚举

用**子串**归类，且**先判 failed 再判 done**——兜住 `Unsuccessful` 这类既含 `success` 又表失败的歧义串：

```python
_DONE = ("success", "finished", "complete", "done")
_FAIL = ("unsuccess", "fail", "error", "cancel", "stopped", "abort")

def is_failed(status): return any(h in (status or "").lower() for h in _FAIL)
def is_done(status):   return any(h in (status or "").lower() for h in _DONE)

# 轮询循环
import time
while True:
    st = query_task(api, task_id, fs_id)
    s = st.get("status", "")
    if is_failed(s):
        print("FAILED:", s, st.get("failed_count"), "个对象失败"); break
    if is_done(s):
        print("DONE:", st.get("bytes_done"), "/", st.get("bytes_total")); break
    time.sleep(60)
```

## 7. 取消任务

```python
api.cancel_data_flow_task(vepfs.CancelDataFlowTaskRequest(
    file_system_id=fs_id,
    data_flow_task_id=str(task_id),
))
```

## 8. 常见错误码

| 错误码 | 原因 |
|--------|------|
| `InvalidParameter.BucketMaybeNotExist` | TOS 桶不存在，多半漏填桶名。确认 `data_storage` 是有效裸桶名 |
| `InvalidParameter.BucketName` | 桶名格式非法：`data_storage` 里带了 `tos://` 前缀或多余斜杠 |
| `InvalidParameter.SourceStoragePrefix` | 路径前缀非法：`data_storage_path`/`sub_path` 未写成 `/a/b/`（首尾带斜杠） |
