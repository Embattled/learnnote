# Microsoft.PowerShell.Management

PowerShell 的 Microsoft.PowerShell.Management module, 主要用于对 Windows 进行管理


# Service 服务管理

对常驻系统服务进行各种管理, 有点类似于 linux 下的 systemctl, 但是分为了很多个命令

## Start-Service

## Get-Service

获取服务的运行状态



## Set-Service

对服务进行启动配置, 例如开机自启等

```Powershell
Set-Service
两种服务名称选定方式

   [-Name] <String>
   [-InputObject] <ServiceController>

共同配置参数:
   [-DisplayName <String>]
   [-Credential <PSCredential>]
   [-Description <String>]
   [-StartupType <ServiceStartupType>]
   [-Status <String>]
   [-SecurityDescriptorSddl <String>]
   [-Force]
   [-PassThru]
   [-WhatIf]
   [-Confirm]
   [<CommonParameters>]
```

- StartupType  启动方式
  - Automatic   : 随系统自动启动, 如果某个服务被设置成自动启动, 那么该服务所以来的基础服务也将会自动被变更为自动启动
  - AutomaticDelayedStart : 略微延迟的自启动
  - Disabled    : 屏蔽服务, 不能被应用或者用户启动
  - Manual      : 手动启动, 该服务不会自动启动, 但是可以被用户或者应用触发启动