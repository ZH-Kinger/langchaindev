# ossutil 2.2.2 `du` 输出格式取证 — 修 estimate_source 解析 bug

取证方式【实测】：SSH → bot-new(8.222.149.27) → `docker exec aiops-bot` → 走真实代码路径
`core.ssh_transfer.engine_ssh.run(...)` 遥控 SGP(43.98.203.59) 跑 `ossutil du`。只读，未改任何状态。

- 版本【实测】：`ossutil version` → `2.2.2`
- 目标前缀：`oss://wuji-data-tran/ossutil_output/`（刚验收首单，约 22MB / 3 对象）

## 1. 逐字原始输出（repr 保留 \r\t\n 与列对齐）

命令：`ossutil du oss://wuji-data-tran/ossutil_output/ 2>&1`，rc=0，err 空。

```
'\r\r\r                                                                      \r'
'storage class \tobject count        \tsum size                      \n'
'----------------------------------------------------------\n'
'Standard      \t3                   \t23208637                      \n'
'----------------------------------------------------------\n'
'total object count: 3                   \ttotal object sum size: 23208637\n'
'\r                                                                      \r'
'total part count:   0                   \ttotal part sum size:   0\n'
'\r\n'
'total du size:23208637\n'
'\n'
'0.501793(s) elapsed\n'
```

排成人读版（列以 TAB 分隔）：

```
storage class    object count         sum size
----------------------------------------------------------
Standard         3                    23208637
----------------------------------------------------------
total object count: 3        total object sum size: 23208637
total part count:   0        total part sum size:   0
total du size:23208637
0.501793(s) elapsed
```

## 2. 各数字在哪一行、措辞逐字

- **字节数（源大小）**：`23208637` = 22.13 MiB。三处都是这个值：
  - 数据行 `Standard\t3\t23208637`（第三列 = sum size）
  - 汇总行 `total object sum size: 23208637`（**冒号+空格**后）
  - 末行 `total du size:23208637`（**冒号无空格**，含 part，此例 part=0 故相等）
- **对象数**：`3`
  - 数据行第二列 `object count`
  - 汇总行 `total object count: 3`（冒号+空格后）
- **单位**：纯字节整数，**无** MB/GB 后缀，**无**千分位逗号。
- **列顺序**：`object count`(第2列) 在 `sum size`(第3列) **前面**（左→右：storage class → object count → sum size）。
- **汇总行措辞逐字**（可锚定的稳定串）：
  - `total object count:` `total object sum size:`
  - `total part count:` `total part sum size:`
  - `total du size:`（注意冒号后**紧跟数字**，无空格）
- 首尾有回车+空格的进度条覆写残留（`\r ... \r`），不影响按串匹配。

## 3. 现有 bug 机理（确认）

现正则 `re.search(r"sum\s*size[^\d]*([\d,]+)", text, re.I)`：
`sum size` **首次命中在表头行** `... object count \tsum size`。表头 "sum size" 之后是
空格→换行→分隔线→换行→`Standard`→TAB→`3`，`[^\d]*` 一路跨行吞到第一个数字 `3`
（数据行的 object count 列），于是把 3 对象误当成 3 字节。这就是「22MB 解析成 3 字节」。
对象数正则 `object\s*count[^\d]*([\d,]+)` 同样先命中表头，跨行吞到 `3`——此例碰巧对，但同样脆。

## 4. 正则修法建议（正则由 dev 写）

锚定汇总行的**唯一措辞串**，别碰表头/数据行：

- **字节数**：锚 `total object sum size:` 抓其后数字 → 稳取 23208637。
  - 备选/兜底：`total du size:`（冒号后可能无空格，故 `[:\s]*` 或 `:\s*`；含 part，通常与 object sum size 相等，纯对象迁移用 object sum size 更贴切）。
- **对象数**：锚 `total object count:` 抓其后数字 → 3。
- 关键点：匹配串里带上 `total` + `object`/`du`，避免命中表头的裸 `sum size`/`object count`；
  数字段用 `[\d,]+` 兼容（虽本例无逗号）；捕获前用 `[:\s]+` 而非 `[^\d]*`（`[^\d]*` 会跨行乱吞，是本 bug 根因）。
- 建议：优先匹配 `total object sum size` / `total object count`，找不到再回退 `total du size`；
  两个都没命中才 ok=False（现有 fail-safe 逻辑保留）。

安全：全程只 `du`/`version`，未做 cp/rm/sync，未改 SGP/OSS/容器任何状态；输出无凭证。
