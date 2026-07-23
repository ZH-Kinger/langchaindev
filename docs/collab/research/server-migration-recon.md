# 服务器迁移只读侦察报告

> 只读侦察，未改动/未搬运任何东西。产出供 planner 出迁移方案。
> 侦察时间：2026-07-10 上午（北京时）。
> 可信度标注：【实测】=本次 SSH 只读反查所得；【文档】=CLAUDE.md/notes 既有事实；【推测】=未证实。

## 0. 最重要的一条结论（先看这个）

**【实测】新机 `bot-new`（8.222.149.27）不是空机——它已经在跑一套 aiops-bot！**

- 新机当前运行：`aiops-bot`（langchaindev-bot:latest，Up 8 天）+ `redis-agents`（redis:8，Up 8 天），端口 8088/6379 已被占用。
- 新机已有 `/root/langchaindev`（Jul 1 的代码快照，非 git repo，`.deployed_commit` 为空）。
- 新机 bot `/health` = `crypto ok / dsw_api ok / feishu ok / jira not_configured / prometheus ok / redis ok`——**基本能跑**，但是**旧版本**（Jul 1 代码，比旧机 HEAD 落后：缺 ssh_transfer T0–T6、缺 paramiko；CLAUDE.md 23401B vs 旧机 30460B）。
- 新机还跑着**别的无关服务**：`wuji-review-server`（healthy）、`marzneshin-*`（VPN/代理面板 Marzneshin）、三个 30GB 的 `wuji-openwam` 镜像。**这是台共享繁忙机，不是专用机。**

**待 dev/用户确认**：这套 Jul-1 的旧 bot 是不是就是本次迁移的目标底座？迁移动作实质 = **把新机这套升级到当前 HEAD + 补全 .env + 配好回调**，而非从零装机。若要「整体覆盖重来」需另议（会动到别人的共享机）。

---

## 1. 两机系统对比

| 项 | 旧机 bot-server (115.191.2.86) | 新机 bot-new (8.222.149.27) |
|----|------|------|
| OS | CentOS Stream 9, kernel 5.14 | CentOS Stream 10, kernel 6.12 |
| CPU | 2 核 | 2 核 |
| 内存 | **1.8Gi**（已用 1.2Gi，可用仅 607Mi，swap 2G） | **7.4Gi**（可用 5.1Gi，无 swap） |
| 磁盘 | / 40G，用 17G，**剩 21G** | / 100G，用 65G，**剩 36G**（30GB×openwam 吃掉大头） |
| 时区 | Asia/Shanghai (CST+0800)，NTP 同步 | Asia/Shanghai (CST+0800)，NTP 同步 |
| Docker | （旧版，未查 version） | 29.4.3 |
| Compose | | v5.1.3 |
| git | | 2.52.0 |

【实测】**两机时区已一致（都 Asia/Shanghai）**，无需调整。注意容器内按 UTC 跑、应用调度器按北京时换算——这是应用内逻辑，随镜像走，不受宿主时区影响。
【实测】关键约束印证 CLAUDE.md 记忆：**旧机 1.8G 内存扛不住 `docker compose up -d --build`（会 OOM），新机 7.4G 可以 build。**

---

## 2. Docker 全景

### 旧机
- 运行容器 2 个：`aiops-bot`（langchaindev-bot）、`redis-agents`（redis:latest）。**没有独立 nginx/mysql 容器**（见下，nginx/mysql 是宿主机进程，与 bot 无关）。
- 镜像：`langchaindev-bot:latest`(3.77GB) + `:backup-2026-05-22-1504`(3.45GB) + `redis:latest`(204MB)。
- 网络：bot 与 redis 都用 **`network_mode: host`**（compose 里写死）。
- 卷：redis-agents **无任何 volume 挂载**（`Mounts: []`）——见第 9 节，重大隐患。另有一组 `server-side_*` 卷（grafana-data/prometheus-data/redis-data/server-data）是**已停用的 server-side compose 项目遗留**，当前无容器在用，与 bot 无关。

### 新机（已存在的部署）
- 运行容器 5 个：`aiops-bot`、`redis-agents`(redis:8)、`wuji-review-server`、`marzneshin-marznode-1`、`marzneshin-marzneshin-1`。
- 镜像一堆（含 3×30GB 的 wuji-openwam、redis:7、redis:8、wuji-review-server 多 tag、marzneshin 等）。`langchaindev-bot:latest` 3.7GB 已构建好。
- **新机 compose 比旧机更完善**（见第 3 节）。

---

## 3. compose / 构建 对比（关键差异）

**旧机 `docker-compose.yml`**（只有 bot 一个 service，redis 是手工起的孤儿容器）：
```yaml
services:
  bot:
    build: .
    container_name: aiops-bot
    restart: unless-stopped
    env_file: .env
    network_mode: host
    volumes: [ .:/app, ./logs, ./sessions, ./vector_db, ./models ]
```
- redis-agents **不在 compose 里**，是裸 `docker run redis --requirepass ...` 起的（Cmd 里硬编码密码，无 volume）。

**新机 `docker-compose.yml`**（更好，redis 纳入 compose + 持久化）：
```yaml
services:
  redis:
    image: redis:8
    container_name: redis-agents
    network_mode: host
    command: ["redis-server","--requirepass","${REDIS_PASSWORD}","--appendonly","yes"]
    volumes: [ redis-data:/data ]
  bot:
    build: .
    depends_on: [redis]
    ... (同旧机 volumes)
volumes: { redis-data: }
```
- **新机已解决旧机的 redis 不持久化问题**：AOF 开启 + 命名卷 `langchaindev_redis-data`，密码走 `${REDIS_PASSWORD}` 环境变量（不再硬编码进 Cmd）。**建议迁移采用新机这版 compose。**

**Dockerfile**（两机基本一致，`python:3.11-slim` + 清华源 + CPU 版 torch + 注释掉 langchain-chroma 导致 RAG 不可用/bot 不受影响）。旧机 Dockerfile 1365B、新机 1343B，差异需 dev 核（可能是清华源/注释行微调）。**镜像重建要 `up -d --build`（新机内存够）。**

**restart 策略**：两边都 `unless-stopped`。**依赖**：新机 bot `depends_on: redis`（旧机无，靠手工顺序）。

---

## 4. 项目目录运行时数据（旧机，迁移必搬的非代码部分）

【实测】旧机 `/root/langchaindev` **总共只有 24M**，运行时数据极小：

| 路径 | 大小 | 说明 |
|------|------|------|
| `.env` | 12K | **必搬**，配置核心（10405 字节，83 键） |
| `.deployed_commit` | 4K | 部署脚本用的游标，可重建 |
| `vector_db/` | 4K | **空**（RAG 已禁用，langchain-chroma 被注释） |
| `models/` | 12K | **几乎空**（无 HuggingFace 模型缓存，RAG 禁用不需要） |
| `sessions/` | 4K | 空（rag 模式才用 FileChatMessageHistory，bot 不用） |
| `data/` | 12K | 小（k8s_docs 种子文本） |
| `logs/` | 20M | 运行日志，**不必搬**（重建即可） |

**结论**：项目目录里**唯一必搬的运行时数据就是 `.env`**（12K）。vector_db/models/sessions/data 都空或可重建，无 GB 级数据需要 rsync。代码走 git（当前分支 `feat/oss-perm-level-selective`，仓库在开发机，非服务器）。

**旧机 `/root/langchaindev` 非 git repo**（`git archive HEAD`→scp→解压覆盖 的部署模式，与 CLAUDE.md 一致）。旧机目录还有 `.env.bak` / `.env.bak.1783316374` / `docker-compose.yml.bak` 等备份文件（不必搬）。

---

## 5. .env 键名清单与差异（只键名，无值）

- 【实测】旧机 **83 键**，新机 **67 键**。新机是旧机的**严格子集**（无额外键）。
- **旧机有、新机缺的 16 键**（迁移必补，都是新功能）：
  ```
  DATASET_DASHBOARD_APP_TOKEN / _ENABLED / _INTERVAL_HOURS / _TABLE_ID
  GPU_DIST_BASE_URL / GPU_DIST_ENABLED / GPU_DIST_TOKEN
  MGW_USER_ID
  SGP_SSH_KEY_ENC        ← SSH 迁移链私钥(Fernet 密文)，密钥类
  SSH_TRANSFER_ENABLED
  TRANSFER_OSS_ROLE
  VEPFS_CONFLICT_POLICY_DEFAULT / _DATAFLOW_ENABLED / _FILE_SYSTEM_ID / _FILE_SYSTEM_IDS / _REGION
  ```
- **密钥类键**（名字含 AK/SECRET/KEY/TOKEN/PASSWORD/ENC，迁移时值不可泄漏）：
  `ALIYUN_ACCESS_KEY_ID/SECRET`、`PAI_DSW_ACCESS_KEY_ID/SECRET`、`BOT_CREDS_ENCRYPTION_KEY`(Fernet)、`FEISHU_APP_SECRET`、`FEISHU_VERIFICATION_TOKEN`、`REDIS_PASSWORD`、`JIRA_PAT`、`GRAFANA_API_KEY`、`OPENAI_API_KEY`、`EDGE_API_KEY`、`TOS_ACCESS_KEY/SECRET`、`GPU_DIST_TOKEN`、`SGP_SSH_KEY_ENC`、多个 `CAPACITY_BITABLE_*_TOKEN`/`DATASET_DASHBOARD_APP_TOKEN` 等。
- **⚠️ 陷阱：键名相同 ≠ 值都在。** 新机 `FEISHU_CHAT_ID` 键存在但**值为空**（日志实证：`FEISHU_CHAT_ID 未配置，跳过集群 MFU 早报`、`早报已发送（0 位用户）`）。新机 jira 也 `not_configured`。**最稳妥：直接整份拷贝旧机 `.env` 覆盖新机**（旧机才是完整生产配置），而不是只补 16 个缺键。

---

## 6. 凭证 / 密钥文件（只记路径/权限/字节数，未看内容）

| 文件 | 路径 | 权限 | 大小 | 说明 |
|------|------|------|------|------|
| SGP SSH 私钥 | `/root/bot_sgp_rsa`（**在宿主机 /root，不在 langchaindev 目录**） | 600 | 3243 B | ssh_transfer 用；但**容器实际用的是 `.env` 里的 `SGP_SSH_KEY_ENC`（Fernet 密文）**，这个明文文件是生成密文的源，迁移时可一并搬（或只搬 .env 里的密文即可） |
| root authorized_keys | `/root/.ssh/authorized_keys` | 600 | 206 B | |
| known_hosts | `/root/.ssh/known_hosts` | 600 | 1915 B | |

无额外证书/pem。**结论：密钥面只需 `.env`（含 SGP_SSH_KEY_ENC）+ 可选 `/root/bot_sgp_rsa`。**

---

## 7. 定时任务（系统级）

【实测】**旧机没有任何应用相关的系统级 cron/timer。**
- `crontab -l` = 空（NO_CRONTAB）。
- `/etc/cron.d` 只有系统默认 `0hourly`；cron.daily/hourly 空/默认 `0anacron`。
- systemd timers 全是 OS 自带（tmpfiles-clean / sysstat / dnf-makecache / logrotate）——无一与 bot 相关。

**结论**：所有业务定时逻辑（Jira 轮询 2min / 实例检查 5min / 早报 9:00 / 容量巡检 / OSS 权限推送 / 数据流动对账 等）都是 `core/dsw_scheduler` 在 bot 进程内起的后台线程，**随容器迁移自动带走，无需单独搬 cron**。新机日志已见调度器在跑（`[Scheduler] 早报已发送`）。**无系统级定时任务需要迁移。**

---

## 8. 监听端口 / 对外服务

### 旧机 `ss -tlnp`
- `:8088` python（bot，host 网络直接监听）
- `:6379` docker-proxy（redis-agents）
- `:80` **nginx**（宿主机进程）
- `:22` sshd
- `:3306` + `:33060` **mysqld**（宿主机进程）

【实测】**nginx 不代理 bot**：`/etc/nginx/nginx.conf` 只有默认 `server_name _; listen 80;`，无 `proxy_pass`、无 8088 相关配置。**Feishu 回调是直接打 `http://115.191.2.86:8088/feishu/event`，不经 nginx。** nginx 和 mysql 是这台机上**别的用途**，与 bot 无关，不迁移。

### 新机 `ss -tlnp`
- `:8088` python（已有 bot）+ `:6379` redis（已有）。80/3306 未监听。

**结论**：迁移后新机 bot 同样在 `:8088` 直接对外。**回调 URL 必须从旧 IP 改到新 IP**（见第 11 节风险）。

---

## 9. Redis（重点权衡）

| | 旧机 redis-agents | 新机 redis-agents |
|---|---|---|
| 镜像 | redis:latest | redis:8 |
| 起法 | 裸 `docker run`，**Cmd 里硬编码密码**，不在 compose | compose 管理，`${REDIS_PASSWORD}` 注入 |
| 持久化 | **无 volume（`Mounts: []`）**，dump.rdb 在容器可写层 `/data`（79KB），`appendonly no`，RDB save 开 | **命名卷 `langchaindev_redis-data` + `appendonly yes`（AOF）** |
| DBSIZE | **100 键** | 26 键 |
| bot 连接 | `REDIS_HOST=127.0.0.1:6379 DB=0`，有密码（host 网络直连） | 同 |

【实测】**旧机 redis 数据不持久化**——容器一旦 recreate 就丢（当前 100 键活在容器层）。新机已修好（AOF+命名卷）。

**Redis 数据搬不搬的权衡**（存的是：STS 缓存、会话历史、job 状态机 transfer/cpfs/vepfs/ssh:transfer/bkt、去重键、调度器 ticket 状态、容量快照、mfu 快照、gpu 分布快照）：
- **搬**：在途迁移任务（tr-/sgp-/cpfs-/vepfs- 等）状态连续，重启续跑、进度不丢；调度器 ticket 状态延续。
- **不搬**：新机从零开始——**在途迁移任务会丢进度记录**（对账/refresh 无 Redis 底可自愈），STS 缓存重新 AssumeRole（无害，会自动重建），会话历史清空，去重键清空（极短窗可能重复处理一次事件）。
- **建议**：数据量极小（旧机 100 键/79KB）。**若迁移时点有在途大迁移任务，值得搬**（`redis-cli --rdb` dump 或 `MIGRATE`/`COPY`）；否则可不搬，让新机自然重建（STS/ticket 会自愈，仅丢在途 job 进度显示）。**待 dev/用户确认迁移时点有无在途任务。**
- ⚠️ 安全：**旧机 redis 密码硬编码在容器启动 Cmd 里**（`docker inspect` 可见明文，弱口令），已列入待轮换清单（见 MEMORY 泄漏事件）。新机改用 env 变量注入，较好；建议迁移时顺手轮换该密码。

---

## 10. 宿主机额外依赖

【实测】
- 新机：有 `rsync`、`nc`、`python3`、`git`、`docker`；**无 `ossutil`、无宿主 `paramiko`**。
- **这不影响 bot**：`paramiko` 在容器内（旧机容器 paramiko 5.0.0，pip 装入运行容器，**未烤进镜像**——recreate 会丢，见 MEMORY）；ssh_transfer 的 `ossutil`/`rsync` 实际在 **SGP/泰国远端机**上跑，不在 bot 宿主机。确认无误。
- ⚠️ **新机容器缺 paramiko**（`ModuleNotFoundError`，因新机镜像是 Jul-1 建的、早于 ssh_transfer）。新机 `requirements.txt` 也缺 `paramiko>=3.4.0` 行。**升级到当前 HEAD 必须 `up -d --build` 重烤镜像**，把 paramiko 固化进去（新机 7.4G 内存扛得住 build）。
- 新机 `/root` 有 `ks3util`（金山云工具）、marzneshin 等他人资产，迁移时勿动。

---

## 11. 外部端点连通性（从新机实测，逐个探）

| 目标 | 端口 | 结果 | 用途 |
|------|------|------|------|
| open.feishu.cn | 443 | ✅ OK | 飞书 API |
| sts.cn-hangzhou.aliyuncs.com | 443 | ✅ OK | 阿里 STS |
| ecs.cn-hangzhou.aliyuncs.com | 443 | ✅ OK | 阿里 ECS |
| **43.98.203.59** | **22** | **✅ OK** | **SGP SSH（ssh_transfer 迁移链关键）** |
| jira.wuji.tech | 443 | ✅ OK | Jira |
| dashscope.aliyuncs.com | 443 | ✅ OK | 云端 LLM（OPENAI_BASE_URL） |
| tos-cn-shanghai.volces.com | 443 | ✅ OK | 火山 TOS |
| workspace-...-cn-hangzhou.log.aliyuncs.com | 443 | ✅ OK | 阿里 SLS（PROMETHEUS_URL 实为 SLS 端点） |
| nas.cn-hangzhou.aliyuncs.com | 443 | ✅ OK | 阿里 NAS/CPFS DataFlow |
| **10.0.10.54** | **3000** | **❌ FAIL** | **Grafana（内网 IP，新机不通）** |

**结论**：**除 Grafana 内网 IP 外，所有外部依赖新机都通，含关键的 SGP:22。** Grafana(`GRAFANA_URL=http://10.0.10.54:3000`)是私网地址、新机不在该网段——**待 dev 核实 bot 是否运行期真调 Grafana**（若只用于拼 dashboard 链接则无碍；若渲染图走它则新机会失败）。`GRAFANA_API_KEY` 在 .env。

---

## 12. 必搬清单（给 planner）

| 项 | 搬法 | 大小 | 备注 |
|----|------|------|------|
| 代码 | `git archive HEAD`→scp→解压（deploy.ps1 现成流程），或直接把新机升级到当前 HEAD | ~24M | 仓库在开发机；当前分支 feat/oss-perm-level-selective，需确认部署哪个 ref |
| **`.env`** | scp 整份覆盖新机（含 16 缺键 + 补 FEISHU_CHAT_ID 等空值） | 12K | **唯一必搬的运行时配置**；密钥值不落任何日志/报告 |
| `/root/bot_sgp_rsa` | 可选搬（.env 的 SGP_SSH_KEY_ENC 已含密文，够用） | 3.2K | 权限 600 |
| Redis 数据 | 可选（见第 9 节权衡）——有在途任务才搬 | 79KB/100键 | 新机已 AOF 持久化 |
| vector_db/models/sessions/data | **不搬**（空/可重建，RAG 禁用） | ~0 | |
| logs | **不搬** | 20M | 重建 |
| 镜像 | **新机 `up -d --build` 重建**（含 paramiko），不用 `docker save` | 3.7G | 新机 7.4G 内存能 build；旧机不能 |

---

## 13. 迁移方式建议（给 planner）

1. **镜像**：新机 `docker compose up -d --build`（7.4G 内存够，且能把 paramiko 固化进镜像，根治「recreate 丢 paramiko」）。**不建议 `docker save|load`**（旧镜像缺 paramiko，且体积大）。
2. **数据**：`.env` 用 scp；vector_db/models 无需 rsync（空）。Redis 视在途任务决定（多半不必搬）。
3. **compose**：**采用新机现有版本**（redis 入 compose + AOF + 命名卷 + `${REDIS_PASSWORD}` + depends_on），比旧机裸跑 redis 更健壮。
4. **升级而非重装**：新机已有 Jul-1 底座，最小动作 = 拉当前 HEAD 代码覆盖 + 覆盖 .env + `up -d --build` + 配好回调。

---

## 14. 风险 / 依赖（重点标出）

- 🔴 **飞书回调地址绑死旧 IP**：Feishu 事件订阅回调是 `http://115.191.2.86:8088/feishu/event`（直连、不经 nginx）。**迁移后必须在飞书开放平台把事件订阅回调 URL 改成新机 `http://8.222.149.27:8088/feishu/event`**，否则新机收不到事件。**待 dev/用户确认回调地址在哪配（飞书开放平台后台）+ 是否要 HTTPS/域名。**
- 🔴 **自引用 URL 硬编码旧 IP**：`GPU_DIST_BASE_URL=http://115.191.2.86:8088`（GPU 分布 HTML 页链接）指向旧机。迁移后**必须改成新机 IP**，否则飞书卡片里的「GPU 分布」链接仍指旧机。检查 .env 里其它 `*_BASE_URL`。
- 🟠 **双跑期（用户要求旧机暂不停）**：
  - **飞书事件只能被一台消费**——回调 URL 只能指向一台。指新机 = 旧机静默（收不到事件，但调度器早报/巡检仍会推，可能重复推送到同一群）；指旧机 = 新机纯待命。**两台不能同时收事件。** 尤其早报/容量巡检/OSS 权限推送这类调度器主动推送，**双跑会向同一 `FEISHU_CHAT_ID` 群重复推**——建议双跑期把其中一台的调度器开关/CHAT_ID 关掉。
  - **各自 Redis 不共享**：job 去重键、STS 缓存、ticket 状态各存各的。同一迁移任务若在两台都发起会各建各的（job_id 按天 hash，可能双跑双任务）。
- 🟠 **Grafana 内网不通**（第 11 节）：待核实运行期是否真调。
- 🟠 **新机是共享繁忙机**：还跑 wuji-review-server、marzneshin(VPN 面板)、30GB×3 openwam 镜像，磁盘只剩 36G。`up -d --build` 会再吃几 G，需留意磁盘。**勿动他人容器/镜像。**
- 🟡 **Redis 弱口令硬编码**（旧机 Cmd 明文，已在待轮换清单）：迁移顺手轮换。
- 🟡 **旧机 bot 当前 jira `error HTTP 500`**（degraded），新机 jira `not_configured`——Jira 连接本身有问题，与迁移无关但迁移后需复核 JIRA_PAT/JIRA_URL。
- 🟡 **端口冲突**：新机 8088/6379 已被现有 bot/redis 占——因为就是同一套要升级的部署，非冲突；若要并行第二套则冲突。

---

## 15. 待 dev/用户确认清单

1. 新机那套 Jul-1 的 aiops-bot 是不是迁移目标底座？（→ 升级 vs 覆盖重装）
2. 部署哪个 git ref？（当前分支 `feat/oss-perm-level-selective` 有大量未提交/未部署改动，HEAD=4857f24）
3. 飞书事件订阅回调 URL 在哪配、要不要切到新机 IP、要不要 HTTPS/域名（现在是裸 http+IP+8088）。
4. Redis 数据搬不搬（取决于迁移时点有无在途迁移任务）。
5. Grafana(10.0.10.54:3000) 运行期是否必需（新机不通）。
6. 双跑期怎么避免两台同时推送/消费飞书事件（关一台调度器？）。
7. 迁移时是否顺手轮换 Redis 密码 + 待轮换凭证清单里其它项。
