"""SSH 迁移链：杭州 OSS →(CEN)→ 新加坡 OSS(挂载盘) →(rsync)→ 泰国服务器。

bot 用 paramiko SSH 到新加坡 ECS，遥控两段 shell（段1 ossutil、段2 rsync），bot 自己不搬数据。
照搬 core/transfer 的状态机 + Redis job + run_to_completion 轮询骨架，把"提交云 API + poll 云
status"换成"SSH 起 nohup 后台命令 + SSH 轮询 pid/rc marker 文件"。独立 Redis 命名空间
ssh:transfer:*，job 前缀 sgp-。
"""
