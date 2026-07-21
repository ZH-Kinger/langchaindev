"""PFS↔PFS 跨云直传（vePFS↔CPFS 三段链编排）。

物理上无 vePFS↔CPFS 直连：每个 PFS 只能对接本厂对象存储。故走 3 段：
    源 PFS --沉降(Export)--> 源云对象存储 --跨云迁移--> 目的云对象存储 --预热(Import)--> 目的 PFS

本包是**加法式上层编排**，串已有三段引擎（core/vepfs_dataflow、core/transfer、core/cpfs_dataflow），
不改任何现有引擎。参照 core/ssh_transfer 的多段链六件套 + 段级"跳过已成功段"retry。
"""
