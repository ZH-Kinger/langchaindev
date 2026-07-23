# 服务器迁移 Runbook（旧机 bot-server → 新机 bot-new）

> 面向 **dev** 的分步骤执行计划。事实源：`docs/collab/research/server-migration-recon.md`（researcher 侦察）+ 用户已拍板 4 决定。
> 本文只规划、不执行。**凡标 🔐 的步骤需用户在场并明确授权**（写生产 / 删容器 / 改飞书回调）。
> 编写：planner，2026-07-10。

---

## 0. 一页速览

| 项 | 结论 |
|----|------|
| 旧机 `bot-server` | 115.191.2.86，1.8G，**现网 live**，飞书回调当前指它。全程**不停、不动**，充当天然回滚位。 |
| 新机 `bot-new` | 8.222.149.27，2 核 7.4G，docker29/compose5/git 已装。已跑一套 Jul-1 旧版 langchaindev（要删重装）+ **别人的** wuji-review-server / marzneshin / openwam（**绝不能碰**）。 |
| 部署目标 | 新机 `/root/langchaindev` 全新部署 **当前 HEAD**（brief 记 `7050f65`，含 ssh_transfer 全套 + paramiko；dev 部署时钉死确切 SHA）。 |
| 部署后状态 | 新机 = **热备**，不接飞书事件（回调仍指旧机），**调度器双推必须抑制**（见 §双推抑制）。 |
| 必搬数据 | `.env`（整份覆盖）+ Redis 数据（初拷 + cutover 末次同步）+ **RAG 嵌入模型缓存 `models/model_cache`（~391M）+ 向量库 `vector_db/`**（git 不跟踪，git archive 同步不了，缺了知识库问答离线加载会失败）。代码走 git archive。sessions/data/logs 不搬。 |
| 构建方式 | **`docker compose up -d --build`**（新机 7.4G 扛得住，把 paramiko 固化进镜像，别学旧机 exec pip 临时装）。 |
| cutover（切回调） | **留给用户择时手动执行**：末次 Redis 同步 + 飞书开放平台改回调到新机 IP + 旧机降备。 |

### 需要用户在场/授权的步骤（🔐）
- **Phase C**：`docker compose down` 删新机现有那套 langchaindev（会短暂中断新机上那套旧 bot；不碰别人服务）。
- **Phase E**：`up -d --build`（新机首次写生产容器 + 吃几 G 磁盘）。
- **Phase F / H-1**：Redis 从旧机（live）导出（读生产 + 处理弱口令密码）。
- **Phase H**（cutover）：改飞书开放平台事件回调 URL、旧机降级 —— **全程用户手动**，本 runbook 只给 checklist。

---

## 事实基线（researcher 已核实，dev 执行前默认成立，如遇不符即停）
- 新机 `langchaindev` compose 项目（aiops-bot + redis-agents）与 server-side / marzneshin / openwam **完全独立** → `docker compose down` 只影响这套。
- 新机现有 compose 已规范：`redis:8` 容器名 `redis-agents`、`--requirepass ${REDIS_PASSWORD} --appendonly yes`、命名卷 `redis-data:/data`、bot `depends_on: redis`、均 host 网络（8088/6379）。
- **`docker-compose.yml` 与 `Dockerfile` 不在 git 仓库**（未跟踪、只存在于各服务器）→ `git archive HEAD` **不会覆盖**它们。新机现有的「好版本」compose/Dockerfile 会**原样保留**；只有 `requirements.txt`（已跟踪，含 `paramiko>=3.4.0`）随代码更新 → 触发重新 `--build` 装 paramiko。
- 连通性新机全绿：feishu / 阿里 STS·ECS·SLS·NAS / Jira / dashscope / 火山 TOS / **SGP 43.98.203.59:22**。**仅内网 Grafana 10.0.10.54:3000 不通** → 见风险 R4（不影响运行期）。
- 无系统级 cron/timer 需迁移（业务定时全在 bot 进程内线程，随容器走）。
- 项目目录 24M，运行期必搬 = `.env` + Redis + **`models/model_cache`（~391M）+ `vector_db/`**（后两者 git 不跟踪、被 `.gitignore` 的 `models/`/`*.safetensors`/`vector_db/` 排除，`git archive` 永远同步不了 → **必须单独 scp**，否则知识库问答触发离线嵌入模型加载失败、报连不上 huggingface.co）。sessions/data 空或可重建。

---

## Phase A — 预检（只读，无授权门）

目的：坐实迁移前提，抓出会致命的隐患（尤其 `BOT_CREDS_ENCRYPTION_KEY` 新旧机不一致）。

| 步骤 | 命令要点 | 预期结果 | 验收 |
|------|---------|---------|------|
| A1 两机可达 | `ssh bot-server 'hostname; date'`、`ssh <bot-new> 'hostname; date'`（dev 需先配好新机 SSH 别名/凭证；本 runbook 记为 `bot-new`） | 两机都能登、时区都 CST+0800 | 均返回主机名 |
| A2 **Fernet key 一致性（最关键）** | 分别读两机 `.env` 里 `BOT_CREDS_ENCRYPTION_KEY` 的**值**，本地比对是否**逐字相同**。**只比对、不打印进任何日志/报告/消息**（可用 `ssh bot-server "grep -c '^BOT_CREDS_ENCRYPTION_KEY=<预期值>' /root/langchaindev/.env"` 之类不回显值的手法，或用 md5：`ssh bot-server "grep '^BOT_CREDS_ENCRYPTION_KEY=' /root/langchaindev/.env | md5sum"` 两机对比 md5） | 两机 md5 相同 | **不同 → 立即停**：`SGP_SSH_KEY_ENC`/用户 AK 密文将无法解密，SSH 迁移链与已绑定用户全废。此时决策：以旧机 key 为准（整份覆盖 .env 会带过来，天然一致，见 Phase D），或重新加密所有密文（不推荐）。 |
| A3 端口 | `ssh bot-new "ss -tlnp | grep -E ':8088|:6379'"` | 8088/6379 被现有 aiops-bot/redis-agents 占（正常，Phase C down 后释放） | 记录当前占用进程，Phase C 后应消失 |
| A4 磁盘 | `ssh bot-new "df -h / && docker system df"` | / 剩 ~36G，够 `--build`（再吃几 G） | 剩余 > 10G；否则先清 dangling 镜像（**只清无 tag 悬空层，不动 openwam/marzneshin/review 镜像**） |
| A5 **别人服务快照**（事后核对没动他们） | `ssh bot-new "docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}' | sort > /tmp/pre_migration_ps.txt; cat /tmp/pre_migration_ps.txt"` | 含 wuji-review-server / marzneshin-marznode-1 / marzneshin-marzneshin-1（及其镜像/状态） | 存档此清单；Phase G 结束后再拉一次比对：**这三个必须 Status 不变（未重启、未消失）** |
| A6 当前健康基线 | `ssh bot-new "curl -s http://localhost:8088/health"` | Jul-1 旧 bot 的 /health（crypto/dsw_api/feishu/prometheus/redis ok，jira not_configured） | 记录基线，Phase G 后对比 |

**回滚**：Phase A 全只读，无回滚需求。A2 不一致是**硬停点**，不得进入后续阶段。

---

## Phase B — 部署代码到新机 `/root/langchaindev`（无授权门，只写代码文件不动容器）

**deploy.ps1 硬编码 `$SERVER="bot-server"`**（第 32 行），直接跑会部署到旧机。两条路：

- **推荐（手工 git archive，一次性迁移最省心，不改脚本）**：在开发机仓库目录执行
  ```
  # 1. 钉死要部署的 ref（先确认 HEAD 是含 ssh_transfer 全套的那个 SHA）
  git rev-parse --short HEAD          # 期望 7050f65（或 dev 确认的当前 HEAD）
  # 2. 打包已跟踪文件（天然排除 .git / __pycache__ / .env / 数据目录）
  git archive --format=tar -o %TEMP%\bot-new-deploy.tar HEAD
  # 3. 传到新机
  scp %TEMP%\bot-new-deploy.tar bot-new:/tmp/
  # 4. 新机解压覆盖（/root/langchaindev 非 git 仓库，与旧机部署模型一致）
  ssh bot-new "cd /root/langchaindev && tar -xf /tmp/bot-new-deploy.tar && \
               find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null; \
               rm -f /tmp/bot-new-deploy.tar && echo DEPLOY_EXTRACT_OK"
  ```
- **可选（改造 deploy.ps1）**：把 `$SERVER`/`$REMOTE` 提成 `-Server` 参数（dev 的代码改动，需过审计+用户授权改脚本），再 `.\deploy.ps1 -Server bot-new`。一次性迁移**不建议**为此改脚本，手工法更快且可控。

| 步骤 | 命令要点 | 预期结果 | 验收 |
|------|---------|---------|------|
| B1 钉 ref | `git rev-parse HEAD` | 确切 40 位 SHA，记进迁移记录 | dev 书面确认该 ref = 当前 HEAD |
| B2 archive+scp+解压 | 见上 | 打印 `DEPLOY_EXTRACT_OK` | `ssh bot-new "cat /root/langchaindev/core/ssh_transfer/engine_ssh.py | head -1"` 能读到（证明新代码已落地） |
| B3 确认 requirements 已更新 | `ssh bot-new "grep -i paramiko /root/langchaindev/requirements.txt"` | 命中 `paramiko>=3.4.0` | 命中 → Phase E 的 `--build` 会装它 |
| B4 确认 compose/Dockerfile **未被覆盖** | `ssh bot-new "grep -c 'redis-data' /root/langchaindev/docker-compose.yml"` | ≥1（仍是含 AOF+命名卷的好版本） | 命中 → git archive 没动 compose，符合预期 |

> **陈旧文件说明**：新机原是 Jul-1 快照，overlay 解压只覆盖/新增、**不删** Jul-1 之后仓库里已删除的文件（`.deployed_commit` 为空，无删除 diff）。绝大多数情况无害（Python 不 import 就不执行）。若 dev 想干净，可在 B2 前 `ssh bot-new "rm -rf /root/langchaindev/{core,tools,config,utils,tests}"` 再解压（**只删这些源码目录，保 .env / docker-compose.yml / Dockerfile / logs**）。默认可跳过。

**回滚**：Phase B 只改 `/root/langchaindev` 下代码文件、不动运行中的旧 bot 容器（bind-mount 的代码已换，但容器要到 Phase E restart 才生效）。若要退回：重新解开 Jul-1 备份或不 restart 即可。旧机 live 不受影响。

---

## Phase C — 🔐 删新机现有那套 langchaindev（授权门）

用户决定 #1：删掉新机现有 aiops-bot + redis-agents，全新部署。**不碰 server-side / marzneshin / openwam。**

| 步骤 | 命令要点 | 预期结果 | 回滚 | 验收 |
|------|---------|---------|------|------|
| C1 down 本套 | `ssh bot-new "cd /root/langchaindev && docker compose down"`（**不带 `-v`**：保留命名卷 `langchaindev_redis-data`，Phase F 要往里灌数据；若想彻底清空 Redis 再带 `-v`，本迁移**不带**） | 只停/删 aiops-bot + redis-agents + 其 network；命名卷保留 | `docker compose up -d`（用现有 Jul-1 镜像即可拉回原状） | `ssh bot-new "docker ps --format '{{.Names}}'"` 里 **aiops-bot/redis-agents 消失，wuji-review-server/marzneshin-* 仍在**（对比 A5 快照）；8088/6379 释放 |
| C2 核对没伤及别人 | 对比 A5 的 `/tmp/pre_migration_ps.txt`：三个别人容器 Status 未变 | 三容器仍 Up 原时长 | — | 三容器逐一在跑 |

**注意**：`docker compose down` 作用域 = 当前目录的 compose 项目（项目名默认目录名 `langchaindev`）。researcher 已验证与别人的项目独立。C1 前**务必 `cd /root/langchaindev`**，别在别人项目目录里跑。

---

## Phase D — 🔐 覆盖 .env + 改 host 值 + 校验（授权门：写生产配置）

用户决定 #4 + researcher .env 结论：**整份覆盖**（新机是旧机严格子集、缺 16 键、且有「键名在值空」陷阱 FEISHU_CHAT_ID），不要只补缺键。

| 步骤 | 命令要点 | 预期结果 | 回滚 | 验收 |
|------|---------|---------|------|------|
| D0 备份新机现 .env | `ssh bot-new "cp /root/langchaindev/.env /root/langchaindev/.env.bak.premigrate"` | 旧 .env 存底 | 覆盖出问题时 `cp .env.bak.premigrate .env` | 备份文件存在 |
| D1 **整份搬旧机 .env → 新机**（走 scp，密钥值不落日志） | server↔server 直传或经本地中转，二选一：<br>① 直传：`scp bot-server:/root/langchaindev/.env bot-new:/root/langchaindev/.env`（若 dev 机能同时直连两机）<br>② 中转：`scp bot-server:/root/langchaindev/.env %TEMP%\env.tmp` → `scp %TEMP%\env.tmp bot-new:/root/langchaindev/.env` → **`del %TEMP%\env.tmp`（务必删中转副本）** | 新机 .env = 旧机完整 83 键 | 见 D0 | `ssh bot-new "grep -c '=' /root/langchaindev/.env"` ≈ 旧机键数；`ssh bot-new "grep -E 'SGP_SSH_KEY_ENC|MGW_USER_ID|SSH_TRANSFER_ENABLED' /root/langchaindev/.env | cut -d= -f1"` 三键在场（**只看键名不看值**） |
| D2 **改 host 值 `GPU_DIST_BASE_URL`** | 旧机自引用 `http://115.191.2.86:8088` → 热备期建议**留空或指新机 IP**。热备期用户不点新机链接，可先留空；cutover 时置新机：`http://8.222.149.27:8088`。dev 用 sed 就地改：`ssh bot-new "sed -i 's#^GPU_DIST_BASE_URL=.*#GPU_DIST_BASE_URL=#' /root/langchaindev/.env"`（热备留空版） | 不再指旧机 IP | 见 D0 | `ssh bot-new "grep '^GPU_DIST_BASE_URL=' /root/langchaindev/.env"` 为空或新机 IP |
| D3 核对无其它硬编码旧 IP | `ssh bot-new "grep -n '115.191.2.86' /root/langchaindev/.env"` | **无输出**（除 GPU_DIST_BASE_URL 已处理外，应无其它旧 IP） | — | 无残留旧 IP。（代码侧只有 settings.py:33 一处**注释**含旧 IP，非运行值，无需改） |
| D4 **再次校验 Fernet key**（承 A2） | 整份覆盖后新机 `BOT_CREDS_ENCRYPTION_KEY` = 旧机值（天然一致） | 与旧机 md5 相同 | — | 与旧机 md5 一致 → SGP_SSH_KEY_ENC/用户 AK 可解密 |
| D5 **写入双推抑制开关**（见 §双推抑制，热备必做） | 按选定方案在新机 .env 追加/改写抑制项 | 热备期调度器不向用户群双推 | 见 D0 | 见 §双推抑制的验收 |

> ⚠️ **陷阱**：旧机 `.env` 里 `REDIS_PASSWORD` 会随整份覆盖带到新机，与新机 compose 的 `${REDIS_PASSWORD}` 注入一致（用户决定 #4：弱口令保持现状不轮换）。**确认覆盖后新机 `REDIS_PASSWORD` = 旧机同值**，否则 Phase E 起来的新 redis 密码变了、bot 连不上、Phase F 导入也对不上密码。验收：`ssh bot-new "grep '^REDIS_PASSWORD=' /root/langchaindev/.env | md5sum"` == 旧机同项 md5。

---

## Phase E — 🔐 `up -d --build`（授权门：写生产容器 + 烤 paramiko）

| 步骤 | 命令要点 | 预期结果 | 回滚 | 验收 |
|------|---------|---------|------|------|
| E1 构建并起 | `ssh bot-new "cd /root/langchaindev && docker compose up -d --build"` | 重新构建镜像（装 paramiko）、起 redis-agents(redis:8, AOF, 命名卷) + aiops-bot；**改了 .env 用 `up -d`（force-recreate 语义）能重载 env_file**（对照 MEMORY「改 .env 必须 recreate 不能 restart」——这里是 up --build 会 recreate，OK） | `docker compose down` + `docker compose up -d`（回 Jul-1 镜像）；旧机始终 live 兜底 | 构建无 error、两容器 Up |
| E2 容器内验 paramiko 固化 | `ssh bot-new "docker exec aiops-bot python -c 'import paramiko; print(paramiko.__version__)'"` | 打印 ≥3.4.0 | — | **成功 import**（证明烤进镜像，非临时 pip；recreate 不再丢） |
| E3 磁盘复查 | `ssh bot-new "df -h /"` | 剩余仍 > 5G | 清悬空镜像 `docker image prune`（**只清 dangling**） | 未占满 |

> **为何必须 `--build`**：requirements 新增 paramiko，restart 不装依赖（deploy.ps1 也是这逻辑）。旧机 1.8G build 会 OOM 才 exec pip 临时装（recreate 丢）；新机 7.4G 无此限制，务必 build 固化。

---

## Phase F — 🔐 Redis 数据初拷（旧机 live → 新机）（授权门：读生产）

用户决定 #3：搬 Redis 数据（STS 缓存 / dsw 工单状态 / 在途迁移 job 进度 / 去重键，旧机约 100 键/79KB）。
⚠️ 旧机 redis **无 volume、数据在容器可写层** → 用**导出 RDB → 灌入新机命名卷**的方式。密码走 `-a` 且**不打印密码**。

**正确姿势（RDB 快照法，最稳）**：

| 步骤 | 命令要点 | 预期结果 | 回滚 | 验收 |
|------|---------|---------|------|------|
| F1 旧机 redis 落盘 | 让旧机 redis 生成最新 dump.rdb（在容器内触发 SAVE，避免打印密码：把密码放进容器内 env 或用 `--no-auth-warning` 且从 .env 读）：<br>`ssh bot-server "docker exec aiops-redis容器名 sh -c 'redis-cli -a \"$REDIS_PASSWORD\" --no-auth-warning SAVE'"`<br>（`$REDIS_PASSWORD` 在旧机容器环境里已有则直接引用，**命令行不出现明文**；旧机 redis 容器名以实际为准，recon 记为 `redis-agents`） | 返回 `OK`，`/data/dump.rdb` 更新 | 无（只读旧机+生成快照，不改旧机数据） | `docker exec <redis> sh -c 'ls -l /data/dump.rdb'` 时间戳为刚才 |
| F2 拷出 RDB | `ssh bot-server "docker cp <redis容器>:/data/dump.rdb /tmp/redis_dump.rdb"` 然后 scp 到新机：`scp bot-server:/tmp/redis_dump.rdb bot-new:/tmp/`（或经本地中转，中转副本用完即删） | 新机 /tmp 有 dump.rdb（~79KB） | — | 文件大小与旧机一致 |
| F3 灌入新机命名卷 | 新机 redis 先停（避免 AOF 与导入打架）：`ssh bot-new "cd /root/langchaindev && docker compose stop bot redis"`；把 rdb 放进命名卷并**关掉 AOF 优先或用 rdb 引导**：<br>命名卷路径：`docker volume inspect langchaindev_redis-data`（取 Mountpoint）→ `cp /tmp/redis_dump.rdb <Mountpoint>/dump.rdb`；**若 AOF 开着，redis 启动会优先读 AOF 而非 RDB** → 需临时 `appendonly no` 引导一次或用 `redis-cli --rdb`/`DEBUG RELOAD`。**推荐更干净的替代**：见 F3-alt。 | 数据进入新 redis | 重新 `up -d` 用空卷（放弃搬数据，接受 recon §9「不搬」的自愈路径） | F5 验收 |
| **F3-alt（推荐，绕开 AOF/RDB 优先级坑）** | 不动卷文件，改用**在线 `--pipe`/`RESTORE` 迁移**或 `redis-cli -a ... --rdb` 拉源库 + 逐键 `MIGRATE`/`COPY`。数据量极小（100 键），最稳是 dump→逐键 restore 脚本；或直接：源机 `redis-cli -a $PW --rdb /tmp/x.rdb` 导出，目标 `redis-cli -a $PW --pipe < <(rdb 转 restore 流)`。**dev 择一，核心要求：① 密码不落命令行明文（放容器 env / `--no-auth-warning` 从环境读）；② 导入后 DBSIZE 对得上。** | 同 F3 | 同 F3 | 同 F5 |
| F4 起新机 | `ssh bot-new "cd /root/langchaindev && docker compose up -d"` | redis + bot 起 | — | 两容器 Up |
| F5 验数据到位 | `ssh bot-new "docker exec redis-agents sh -c 'redis-cli -a \"$REDIS_PASSWORD\" --no-auth-warning DBSIZE'"` | 键数 ≈ 旧机初拷时点键数（STS 缓存等短 TTL 键可能已少几个，正常） | — | DBSIZE 与旧机同量级；抽查一个在途 job：`... GET transfer:job:<某id>` 能取到 |

> **新鲜度**：Phase F 是**初拷**——旧机是 live、Redis 持续变（新 STS 缓存、新 dedup 键、在途 job 进度推进）。初拷让新机「大致就位」；**真正 cutover（切回调）时必须再做一次末次同步**（Phase H-1），否则初拷到 cutover 之间的增量丢失（尤其在途迁移 job 进度、期间新工单状态）。
> **安全**：整个 Phase F **不得把 `REDIS_PASSWORD` 明文打进任何命令历史/日志/报告**。用容器内已有的环境变量引用，或 `--no-auth-warning` 配合从 env 读。密码本身弱口令保持现状（决定 #4），迁移不轮换。
> **可跳过判断**：若 cutover 时点**没有在途大迁移任务**，recon §9 指出可不搬（STS/ticket 自愈、仅丢在途 job 进度显示）。但用户已拍板搬 → 照搬；成本极低（79KB）。

---

## Phase G — 热备验收（新机就位但不接飞书事件）

目的：确认新机**技术上完全就绪**，同时**确认调度器双推已被抑制**、飞书回调仍指旧机（新机静默）。

| 步骤 | 命令要点 | 预期结果 | 验收 |
|------|---------|---------|------|
| G1 /health 全绿 | `ssh bot-new "curl -s http://localhost:8088/health"` | crypto ok / dsw_api ok / feishu ok / prometheus ok / redis ok；jira 视 .env（旧机整份覆盖后应 configured，但旧机当前 jira 500，见 R5） | 关键项 ok（对比 A6 基线，应从 Jul-1 not_configured 项转好） |
| G2 settings 各键就位 | `ssh bot-new "docker exec aiops-bot python -c 'from config.settings import settings; settings.print_validate()'"` | 启动自检不报缺 `SGP_SSH_KEY_ENC`/`BOT_CREDS_ENCRYPTION_KEY`/`MASTER_AK` 等（都随整份 .env 到位） | print_validate 无致命缺失告警 |
| G3 SGP 私钥可解密（验 Fernet 一致的实证） | `ssh bot-new "docker exec aiops-bot python -c 'from utils.crypto import decrypt; from config.settings import settings; k=decrypt(settings.SGP_SSH_KEY_ENC); print(k.startswith(\"-----BEGIN\"))'"` | 打印 `True` | **True**（证明新机 Fernet key 能解旧机密文；False → 回 A2 停查） |
| G4 **确认调度器双推已抑制** | `ssh bot-new "docker logs --tail 80 aiops-bot | grep -E 'Scheduler|早报|巡检|对账'"` | 按选定抑制方案：若走「关调度器」→ 无 `[Scheduler] 启动`；若走「env 中和」→ 有启动但 capacity/oss_perm/dataset 未起、FEISHU_CHAT_ID 空致集群早报跳过、reconcile 关 | **见 §双推抑制的验收清单**（这是热备期不打扰用户群的关键） |
| G5 飞书回调仍指旧机（新机不接事件） | 人工确认飞书开放平台事件订阅回调 = 旧机 `http://115.191.2.86:8088/feishu/event`（**未改**）；新机 `/feishu/event` 虽在监听但收不到平台推送 | 用户在群里发消息 → **只旧机响应**，新机日志无 `im.message.receive` | 旧机正常回复；新机日志静默（无入站事件） |
| G6 **别人服务未受影响**（承 A5） | 再拉 `docker ps` 比对 `/tmp/pre_migration_ps.txt` | wuji-review-server / marzneshin-* Status 未变（未被 down/build 波及） | 三容器仍 Up、时长连续 |
| G7 资源竞争体检 | `ssh bot-new "free -h; docker stats --no-stream"` | 7.4G 内存下 aiops-bot + 别人服务共存不吃紧（openwam 是镜像不占运行内存，除非有容器在跑） | 无 OOM、swap 未爆（新机无 swap，尤其注意） |

**回滚（热备阶段）**：任一验收不过 → `docker compose down` 新机这套，旧机 live 完全不受影响（回调本就指旧机），零用户可见影响。修好再来。

---

## Phase H — 🔐 cutover 切回调（**留给用户择时手动执行**，本 runbook 只给 checklist）

> ⚠️ 这是唯一让用户可感知的切换点。**全部动作需用户明确授权并在场**。planner 不排执行时间，由用户择时。

**Cutover checklist（按序）：**
1. **H-1 Redis 末次同步**：重复 Phase F 的导出→导入（旧机 live 到此刻的增量：在途 job 进度、期间新 dsw 工单状态、dedup 键）。做法同 F，密码不外泄。**做完立即进 H-2，缩短「两库不一致」窗口**。
2. **H-2 新机 GPU_DIST_BASE_URL 生效**：把 D2 留空的值改成新机 `http://8.222.149.27:8088`（`sed -i` + `docker compose up -d`（recreate 重载 env）），使飞书卡片「GPU 分布」链接指新机。
3. **H-3 解除双推抑制**：把 §双推抑制里热备期关掉的开关**恢复到旧机生产值**（填回 `FEISHU_CHAT_ID`、按需开 `CAPACITY_MONITOR_ENABLED`/`OSS_PERM_PUSH_ENABLED`/`DATASET_DASHBOARD_ENABLED`/`DATAFLOW_RECONCILE_ENABLED`、若走「关调度器」方案则打开调度器），`up -d` recreate 生效。**此步与 H-4 之间会短暂双推**（旧机还在接事件+推、新机也开始推）→ 建议 H-3 与 H-4、H-5 连续快速执行，或 H-3 放在 H-5 停旧机之后。**推荐顺序：H-4（改回调到新机）→ H-5（停/降旧机）→ H-3（新机开满调度）**，使任一时刻只有一台在推。
4. **H-4 飞书开放平台改回调**（🔐 用户手动）：事件订阅回调 URL `http://115.191.2.86:8088/feishu/event` → `http://8.222.149.27:8088/feishu/event`；卡片回调 `/feishu/card_action` 同理。**核对是否要 HTTPS/域名**（现为裸 http+IP+8088，见开放项 O2）。改完飞书会重新校验回调可达。
5. **H-5 旧机降级为备**：`ssh bot-server "cd /root/langchaindev && docker compose stop bot"`（**先 stop 不 down，留回滚**）。观察新机接管正常后再决定是否 down。旧机作为回滚位保留至少数天。
6. **H-6 cutover 验收**：群里发消息 → **新机**响应（新机日志见 `im.message.receive`）；点「GPU 分布」链接 → 指新机 IP；等到次日 9:00 早报只来一份（无双推）。

**Cutover 回滚**：飞书回调改回旧机 IP + 旧机 `docker compose start bot` → 秒回旧机接管（旧机数据/配置全程未动）。这是本迁移最强的回滚保障。

---

## Phase I — 回滚总纲

| 失败阶段 | 回滚动作 | 用户可见影响 |
|---------|---------|------------|
| A（预检） | 无需回滚（只读）；A2/D4 Fernet 不一致是硬停点 | 无 |
| B（部署代码） | 不 restart 容器即不生效；或解回 Jul-1 备份 | 无（旧机 live） |
| C（删旧套） | `docker compose up -d`（Jul-1 镜像）拉回 | 无（新机那套本就不接生产回调） |
| D（.env） | `cp .env.bak.premigrate .env` + `up -d` | 无 |
| E（build） | `down` + `up -d`（回 Jul-1 镜像）；旧机兜底 | 无 |
| F（Redis 初拷） | 用空卷起新机（放弃搬数据，走自愈）；不影响旧机 | 无 |
| G（热备验收） | `down` 新机这套 | 无（回调仍指旧机） |
| H（cutover） | 飞书回调改回旧机 + 旧机 `start bot` | 秒级切回，短暂消息延迟 |

**贯穿性回滚位**：**旧机 `bot-server` 全程 live、数据/配置零改动** → 任何阶段崩了，最坏就是「新机放弃、继续用旧机」，零数据损失。

---

## §双推抑制（热备期核心，用户决定 #2 的关键推论）

**问题**：新机 bot 一旦 `up -d` 正常启动，`scheduler.start()`（`core/feishu_bot/routes.py:364`，**无条件调用**）会拉起后台线程主动推送到 `FEISHU_CHAT_ID` / 各用户 open_id。旧机也在推 → **同一群/同一用户收到双份**（每日 9:00 早报、集群 MFU、容量巡检、OSS 权限对账、数据流动对账终态卡）。更糟的是 **Jira 轮询线程会主动创建 DSW 实例 + 评论工单**（写操作），两机各存各的 Redis dedup → **可能对同一工单双建 DSW**。

**调度器线程盘点**（`dsw_scheduler.start()`）：

| 线程 | 触发 | 有无 env 开关 | 热备危害 |
|------|------|-------------|---------|
| Jira 轮询（每 2min） | always-on | **无** | 🔴 双建 DSW + 双评论工单（写操作，最危险） |
| 实例检查（每 5min） | always-on | **无**（DSW_IDLE_STOP_ENABLED 只管到期停机，空转提醒不受控） | 🟠 GPU 空转提醒双推给用户 |
| 每日早报 9:00 | always-on | 集群 MFU 报受 `FEISHU_CHAT_ID`/`PROMETHEUS_URL` 门控；**per-user 实例早报按 dsw:ticket 里 open_id 推、不受 FEISHU_CHAT_ID 控** | 🟠 早报双推（尤其搬了 Redis 后 per-user 报按 ticket 推给用户） |
| 数据流动/迁移对账 | `DATAFLOW_RECONCILE_ENABLED`（默认 true） | **有** | 🟠 搬了 Redis 后对在途 job 双推终态卡（各机 NX 闸门独立不跨机去重） |
| 容量巡检 | `CAPACITY_MONITOR_ENABLED`（默认 false） | 有 | 开了才双推 |
| OSS 权限对账推送 | `OSS_PERM_PUSH_ENABLED`（默认 false） | 有 | 开了才双推 |
| 数据集大盘维护 | `DATASET_DASHBOARD_ENABLED`（默认 false） | 有 | 开了才双写 Bitable |

**关键**：危害最大的三个（Jira 轮询、实例检查、per-user 早报）**没有 env 开关**，只靠中和 `FEISHU_CHAT_ID` / 关 opt-in 开关**盖不住**它们。

### 推荐方案 A（首选）：给 `scheduler.start()` 加一道 env 门 —— 需 dev 一处小改代码

**权衡**：热备期我们要新机「尽可能就绪」（能 /health、能被 cutover 秒切），但**完全不产生任何主动副作用**。最干净、最彻底的做法是让热备的新机**根本不启动调度器**（只跑 Flask HTTP + Agent，供 /health 与 cutover 后开启）。

- **改动**（dev 执行，约 3 行，走「查资料→计划→开发→测试→审计」正常流水线，本项即计划）：
  `routes.py` 里 `scheduler.start()` 外包一层 `if getattr(settings, "SCHEDULER_ENABLED", True): scheduler.start()`；`settings.py` 加 `SCHEDULER_ENABLED = os.environ.get("SCHEDULER_ENABLED","true").lower()=="true"`（默认 true，**不改旧机行为**）。
- **热备期**：新机 .env 加 `SCHEDULER_ENABLED=false` → 新机零主动推送/零写 Jira，纯待命。
- **cutover（H-3）**：改 `SCHEDULER_ENABLED=true` + `up -d` recreate → 调度器上线。
- **优点**：一刀切、无遗漏（覆盖 always-on 的三个高危线程）；旧机默认不受影响；可单测（auditor 好审）。
- **代价**：一次小代码改动 + 走一遍审计闸门（用户既有硬规：改动须 auditor 通过才 commit）。

### 备选方案 B（不改代码，纯 env 中和）—— 有残留风险，不完全干净

若用户不想在迁移中夹带任何代码改动，退而求其次靠 .env 值中和，但**盖不住 always-on 的 Jira 轮询与 per-user 早报**：

- `FEISHU_CHAT_ID=`（空）→ 关掉集群 MFU 早报 + 各链对账/巡检回落群频道的推送。
- `DATAFLOW_RECONCILE_ENABLED=false` → 关对账（避免搬来的在途 job 被双推终态卡）。
- `CAPACITY_MONITOR_ENABLED` / `OSS_PERM_PUSH_ENABLED` / `DATASET_DASHBOARD_ENABLED` 保持 false（默认）。
- **Jira 轮询双建 DSW（🔴 最危险）无法用 CHAT_ID 中和** → 只能靠**临时清空新机 .env 的 Jira 凭证/项目键**（如 `JIRA_PROJECT_KEY=` 或 `JIRA_PAT=`）让 `get_gpu_tickets` 取不到工单 → 轮询空转。cutover 时再填回。**这与「整份覆盖 .env」相冲突，属人为制造的临时偏差，易忘记恢复。**
- **per-user 实例早报**：若 Phase F 搬了 `dsw:ticket:*`，9:00 会按 ticket 里的 open_id/feishu_chat_id 推给用户，`FEISHU_CHAT_ID` 空也挡不住 → 残留双推风险。要么热备期**不搬 dsw:ticket 键**（初拷时排除），要么接受这一处窗口。

**结论**：方案 B 遗漏点多（Jira 双建、per-user 早报）、且靠「临时改坏配置再记得改回」很脆。**planner 推荐方案 A**（加 `SCHEDULER_ENABLED` 门），既满足「新机最大程度就绪」又「零打扰用户群」，代价仅一次小改+审计。**是否接受这次代码改动，需 dev 与用户确认**（见开放项 O1）。

### 双推抑制验收（Phase G-4）
- 方案 A：新机日志**无** `[Scheduler] 启动` 行；用户群次日 9:00 早报只来一份（来自旧机）。
- 方案 B：新机日志有 `[Scheduler] 启动` 但 `extra` 不含容量/OSS权限/数据集/对账；日志见 `FEISHU_CHAT_ID 未配置，跳过集群 MFU 早报`；Jira 轮询日志无「创建 DSW」动作。

---

## §风险 / 开放项

**风险（researcher 已识别，runbook 已消化）：**
- **R1 飞书事件单地址消费**：回调只能指一台。热备期指旧机 → 新机静默（靠双推抑制不打扰）；cutover 指新机 → 旧机降备。两机**永不同时接事件**。（Phase G-5 / H-4）
- **R2 调度器双推**：见 §双推抑制（核心）。
- **R3 共享繁忙机**：新机还有 wuji-review-server / marzneshin(VPN) / 3×30G openwam 镜像，磁盘剩 ~36G、无 swap。`--build` 吃几 G，注意 A4/E3/G7 的磁盘与内存体检；**全程不碰别人容器/镜像**（A5/G6 快照对账兜底）。
- **R4 Grafana 内网不通**（10.0.10.54:3000）：**已核实运行期不调用它**——`GRAFANA_URL` 仅用于 `tools/feishu/notify._dashboard_links` 拼**可点链接按钮**，不渲染图；早报/MFU 走 `PROMETHEUS_URL`（实为 SLS 端点，新机 443 通、recon 全绿）。**结论：Grafana 不通不影响迁移与运行**，只是卡片里 Grafana 链接按钮指内网 IP（内网用户可达，外网点不开——本就如此，非新增）。无需处理。
- **R5 Jira 连接**：旧机当前 jira `HTTP 500`（degraded）、新机 Jul-1 `not_configured`。整份覆盖 .env 后新机 jira 配置到位，但**旧机的 500 本身是既存故障**（与迁移无关）→ cutover 后 dev 需复核 `JIRA_PAT`/`JIRA_URL` 有效性；若旧机 jira 一直 500，新机大概率同样 500（Jira 侧问题）。**不阻塞迁移**，记为迁移后待查。
- **R6 Redis 弱口令硬编码**：用户决定 #4 保持现状不轮换。新机 compose 用 `${REDIS_PASSWORD}` env 注入（比旧机 Cmd 明文好）。仍在 MEMORY 待轮换清单，本迁移不动。

**开放项（需 dev 向用户确认后 planner 才能定板 / 或 dev 直接拍）：**
- **O1（最重要）**：双推抑制走**方案 A（加 `SCHEDULER_ENABLED` 门，需一次小代码改+审计）**还是**方案 B（纯 env 中和，有 Jira 双建/per-user 早报残留风险）**？planner 强烈推荐 A。**这直接决定 Phase D-5 / G-4 / H-3 的具体动作，需先定。**
- **O2**：cutover 后飞书回调是否要上 **HTTPS/域名**（现为裸 http+IP+8088）？若要，Phase H-4 需加反代/证书（新机有别人的资产，dev 需确认能否复用/新增，不与他人冲突）。
- **O3**：cutover 时点是否有**在途大迁移任务**？有则 Redis 末次同步（H-1）不可省；无则可简化。
- **O4**：新机 SSH 别名/凭证是否已配好（本 runbook 记为 `bot-new`，dev 需在开发机 `~/.ssh/config` 落别名）。
- **O5**：是否顺手轮换 Redis 弱口令 + MEMORY 待轮换清单其它项？用户已定**本次不轮换 Redis**（决定 #4）；其它项（Master AK / 飞书三凭证 / Fernet / PAI_DSW Secret）是否借迁移窗口一并处理，需用户单独拍（**注意：轮换 Fernet key 会使所有密文失效，与 A2 一致性要求冲突，若轮换须同步重加密——不建议夹带进本次迁移**）。

---

## §执行顺序总览（依赖图）

```
A 预检(只读) ──→ B 部署代码 ──→ C🔐 删旧套 ──→ D🔐 覆盖.env+改host+验Fernet ──→ E🔐 up --build(烤paramiko)
                                                                                      │
                                                              F🔐 Redis初拷 ←─────────┘
                                                                     │
                                                              G 热备验收(含双推抑制确认 + 别人服务未动)
                                                                     │
                                          ┅┅┅┅┅ 热备就绪，长期驻留 ┅┅┅┅┅
                                                                     │
                                          H🔐 cutover(用户择时):H1末次同步→H4改回调→H5停旧机→H3开满调度→H6验收
```

- **串行**：A→B→C→D→E 必须按序（D 依赖 A2 的 Fernet 结论；E 依赖 D 的 .env）。
- **F 可与 G 部分并行**：F（Redis 初拷）在 E 起来后做；G 验收含 F 的结果。
- **A5/G6 别人服务快照对账**贯穿，确保零误伤。
- **双推抑制方案（O1）必须在 D 之前定板**（决定 D-5 写什么）。

---
*(本文档为规划产物，所有写生产/删容器/改回调动作由 dev 在用户授权下执行；planner 不执行迁移。)*
