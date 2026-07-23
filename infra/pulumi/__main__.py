"""aiops-infra 入口：RAM/STS 地基（INFRA-1）+ 创建权限收归（INFRA-1B）。

声明顺序：groups → master → roles → create_governance（依赖 groups + roles）。
执行方式见 README：先 import 存量到 no-op（INFRA-1），再叠加 Deny/Allow（INFRA-1B）。
"""
import pulumi

from ram import groups as ram_groups
from ram import master as ram_master
from ram import roles as ram_roles
from ram import create_governance

# INFRA-1：地基（import-first，preview 应 no-op）
_groups = ram_groups.build()
_master = ram_master.build()
_roles = ram_roles.build()

# INFRA-1B：创建权限收归（有意的新增变更，preview 应仅新增 2 策略 + 挂载）
create_governance.build(_groups, _roles)

pulumi.export("groups", list(_groups.keys()))
pulumi.export("roles", list(_roles.keys()))
pulumi.export("master", _master.name)
