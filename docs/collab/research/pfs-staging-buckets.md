# P1 vePFS→CPFS 直传 — 中转 staging 桶候选（`PFS_STAGING_MAP` 填料）

侦察日期：2026-07-16 ｜ researcher（只读探针）｜ 目标机 `bot-new`（8.222.149.27），容器 `aiops-bot`（Up 18h）
方法：`docker exec aiops-bot python` 调项目现成 client，全程只读（list/describe/detect），未建桶/建 DataFlow/搬数据。

## 链路回顾
```
vePFS --Export沉降--> TOS桶(源侧staging,同vePFS区) --hcs_mgw跨云--> OSS桶(目的侧staging,同CPFS区) --Import预热--> CPFS
       段① 火山 vePFS DataFlow            段② 阿里在线迁移(可跨区)          段③ 阿里 NAS DataFlow
```
- 段①③ 有**同区硬约束**（vePFS↔TOS 同区、CPFS↔OSS 同区）；段②跨云可跨区。
- 配置契约（`core/pfs_transfer/paths.py::_staging_for`）：`PFS_STAGING_MAP` 每个 fs 的 `region` 即该 PFS 与其 staging 桶的共同区域。

---

## 一、火山侧（源）— vePFS 与 TOS staging

**vePFS 文件系统**（`engine_vepfs.list_filesystems`，全部在 **cn-shanghai**，均 Running）：

| fs_id | 名称 | region |
|---|---|---|
| `vepfs-cnshef4f4b647664` | wuji-vepfs（默认 `VEPFS_FILE_SYSTEM_ID`） | cn-shanghai |
| `vepfs-cnsh4bb0c73b50ae` | wuji-vefps-D | cn-shanghai |
| `vepfs-cnsh3d6f174c85e5` | data-infra-D | cn-shanghai |

cn-beijing 查得 0 个。**vePFS 只在 cn-shanghai** → TOS staging 必须 cn-shanghai。【实测】

**TOS 桶**（`get_tos_client().list_buckets()`，共 17 个）。cn-shanghai 的 10 个（可做源侧 staging）：

| 桶名 | region | 备注 |
|---|---|---|
| `data-tran` | cn-shanghai | **推荐**：命名即"data transfer"，与阿里侧 `wuji-data-tran` 成对 |
| `data-sync-b2d` | cn-shanghai | 备选，命名像同步中转 |
| `data-infra-sha` | cn-shanghai | 基建数据 |
| `ego-output` / `eog-pretrain-code` / `las-datastore` | cn-shanghai | 业务桶，不建议占用 |
| `wuji-dc-shanghai` / `wuji-ego-processed` / `wuji-egocentric-data` | cn-shanghai | 业务桶 |
| `ml-platform-auto-created-required-2111674479-cn-shanghai` | cn-shanghai | ML 平台自建，勿动 |

其余 6 个在 cn-guangzhou（umi-*）/ cn-beijing / cn-guangzhou，与 vePFS 不同区，**不可**做源侧 staging。

> 火山 vePFS **无持久 DataFlow 对象**（`submit_task` 直接带 TOS 桶/前缀）→ 源侧 staging 只需一个**可写的同区 TOS 桶**，无需预建绑定。建议在选定桶下用独立前缀（如 `pfs-staging/`）隔离，避免与业务对象混淆。

---

## 二、阿里侧（目的）— CPFS 与 OSS staging（预热要现成 DataFlow）

**CPFS 文件系统**（`engine_nas.list_filesystems`，全部**智算版 `bmcpfs-`**）：

| region | fs_id |
|---|---|
| cn-hangzhou | `bmcpfs-00000ub3ici1dnniit2i0` |
| cn-beijing | `bmcpfs-03001ycys4kpfutqkv8lf` |
| ap-southeast-1 | `bmcpfs-07001v48jdw7tt8jhw0df` |

cn-shanghai 查得 0 个。**CPFS 均不在 cn-shanghai** → 段②必然跨区（cn-shanghai→hz/bj/sing），hcs_mgw 支持，属预期。【实测】

**现有 CPFS↔OSS DataFlow 绑定**（`engine_nas.list_dataflows`，预热 Import 直接可用）：

cn-hangzhou `bmcpfs-00000ub3ici1dnniit2i0`（8 条，全 Running）：

| dataflow_id | OSS 源桶 | source_path | 适合当 staging? |
|---|---|---|---|
| `df-008002b0c9132cc7` | `oss://wuji-data-tran` | `/`（整桶） | **是**，整桶绑定，任意前缀可预热 |
| `df-002dd6a141ae8f81` | `oss://wuji-data-tran` | `/`（整桶） | 同上（重复绑定，任选其一） |
| `df-00ae1aa03dbb9f37` | `oss://wuji-data-tran` | `/`（整桶） | 同上 |
| `df-00dbae98d92d99ef` | `oss://wuji-data-tran` | `/zhangchenhao/` | 子目录级 |
| `df-000bb6a159aba329` | `oss://wuji-bucket-hangzhou` | `/wuji-il/wuji-hand-teleop-data/` | 业务子目录 |
| `df-0032bab83a4996e9` | `oss://wuji-egocentric-processed` | `/result/` | 业务子目录 |
| `df-00bcccb5cb94edc8` | `oss://wuji-egocentric-processed` | `/processed/reprocessed/` | 业务子目录 |
| `df-0015309c347a69c6` | `oss://wuji-bucket-hangzhou` | `/third-party-data/worldengine/` | 业务子目录 |

cn-beijing `bmcpfs-03001ycys4kpfutqkv8lf`：**0 条 DataFlow —— 缺口**。
ap-southeast-1 `bmcpfs-07001v48jdw7tt8jhw0df`：1 条 → `df-07ff53a38df108e2` = `oss://wuji-sing` `/`（整桶）。

**OSS 桶 region 核实**（`detect_bucket_region`，均与所绑 CPFS 同区 ✓）：
`wuji-data-tran`→cn-hangzhou、`wuji-egocentric-processed`→cn-hangzhou、`wuji-bucket-hangzhou`→cn-hangzhou、`wuji-sing`→ap-southeast-1。【实测】

---

## 三、可凑成的完整候选链

### 链 A（推荐，配套最全）— cn-hangzhou 落地
| 环节 | 值 |
|---|---|
| 源 vePFS | `vepfs-cnshef4f4b647664`（wuji-vepfs，默认）@ cn-shanghai |
| 源侧 TOS staging | `data-tran` @ cn-shanghai（建议前缀 `pfs-staging/`） |
| 段② 跨云 | cn-shanghai → cn-hangzhou，hcs_mgw + role `AliyunOSSRole` |
| 目的侧 OSS staging | `wuji-data-tran` @ cn-hangzhou，**dataflow_id `df-008002b0c9132cc7`（整桶）** |
| 目的 CPFS | `bmcpfs-00000ub3ici1dnniit2i0` @ cn-hangzhou |

先跑通就选这条：目的侧整桶 DataFlow 已在（预热任意前缀免建绑定），OSS/TOS 桶命名成对（`data-tran`↔`wuji-data-tran`）。

### 链 B — ap-southeast-1 落地
源 vePFS 同上 → TOS `data-tran`@sha → 跨云 sha→ap-southeast-1 → OSS `wuji-sing`@ap-southeast-1（dataflow `df-07ff53a38df108e2` 整桶）→ CPFS `bmcpfs-07001v48jdw7tt8jhw0df`@ap-southeast-1。

### 链 C — cn-beijing 落地：**缺 DataFlow，暂不可用**（见缺口）

---

## 四、`PFS_STAGING_MAP` 建议填料（给 dev）

配置形状（`paths.py`）：JSON，key=`<scheme>://<fs-id>`；vePFS 值 `{region, tos_bucket, tos_prefix}`，CPFS 值 `{region, oss_bucket, oss_prefix, dataflow_id?}`。覆盖链 A（+B/惯用 fs）：

```json
{
  "vepfs://vepfs-cnshef4f4b647664": {"region": "cn-shanghai", "tos_bucket": "data-tran", "tos_prefix": "pfs-staging"},
  "vepfs://vepfs-cnsh4bb0c73b50ae": {"region": "cn-shanghai", "tos_bucket": "data-tran", "tos_prefix": "pfs-staging"},
  "vepfs://vepfs-cnsh3d6f174c85e5": {"region": "cn-shanghai", "tos_bucket": "data-tran", "tos_prefix": "pfs-staging"},
  "cpfs://bmcpfs-00000ub3ici1dnniit2i0": {"region": "cn-hangzhou", "oss_bucket": "wuji-data-tran", "oss_prefix": "pfs-staging", "dataflow_id": "df-008002b0c9132cc7"},
  "cpfs://bmcpfs-07001v48jdw7tt8jhw0df": {"region": "ap-southeast-1", "oss_bucket": "wuji-sing", "oss_prefix": "pfs-staging", "dataflow_id": "df-07ff53a38df108e2"}
}
```
（`tos_prefix`/`oss_prefix` 我先给 `pfs-staging` 作隔离前缀——**待用户确认**是否愿意用这些业务桶做中转、前缀叫什么；不确定可留空=桶根。）

---

## 五、缺口 / 待确认（交 dev + 用户定）

1. **cn-beijing CPFS 无 DataFlow**：`bmcpfs-03001ycys4kpfutqkv8lf` 无任何 OSS 绑定 → 该区无法预热。要么改走链 A/B，要么请用户在控制台建 OSS→CPFS DataFlow（researcher 不建）。
2. **服务器 .env 缺整个 CPFS 段**：`grep` 未见 `CPFS_REGION/CPFS_FILE_SYSTEM_ID(S)/CPFS_DATAFLOW_ENABLED`。CPFS 侧靠代码默认（`CPFS_REGION=cn-hangzhou`、fs 清单空）。跑 P1 前 dev 需补 `CPFS_FILE_SYSTEM_IDS`、`CPFS_DATAFLOW_ENABLED`（若 orchestrator 依赖）。【实测：.env 无这些键】
3. **段② hcs_mgw 配置**：.env 只有 `MGW_USER_ID=1704065796538912` + `TRANSFER_OSS_ROLE=AliyunOSSRole`，**无 `MGW_REGION`/`MGW_ENDPOINT`/`TRANSFER_BUCKET_MAP`**。跨云段落地 OSS 前 dev 需确认 mgw endpoint/region 是否走默认、以及目的 region（cn-hangzhou/ap-southeast-1）是否被 hcs_mgw 覆盖。
4. **staging 桶写权限/占用**：源侧 TOS `data-tran` 需可写（Export 落盘）；目的侧 OSS `wuji-data-tran` 需被 `AliyunOSSRole` 读到（跨云拉取）+ 已有整桶 DataFlow（✓）。建议用独立前缀避免与业务对象混淆——**待用户拍板用哪个桶/前缀**。
5. **智算版 CPFS Import 需 `ConflictPolicy`**（bmcpfs-）：预热任务参数，dev 侧 orchestrator 已处理（`CPFS_CONFLICT_POLICY_DEFAULT`），确认默认值合适即可。

## 权限/凭证说明
本轮全部只读调用成功（vePFS DescribeFileSystems、TOS ListBuckets、NAS DescribeFileSystems/DescribeDataFlows、OSS GetBucketLocation 均 200），说明容器内现有凭证已具备这些读权限。段②/③的**写**动作（CreateDataFlowTask 预热、hcs_mgw 建 job）本轮未触发，权限待 dry-run/真机验。凭证值一律未打印。
