"""RAM/STS 地基声明模块（INFRA-1）+ 创建权限收归（INFRA-1B）。

导入顺序即声明顺序：groups → master → roles → create_governance。
各 build() 返回声明的资源句柄，供 __main__ 组装与依赖。
"""
