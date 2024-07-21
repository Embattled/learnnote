# Windows PowerShell

* [Windows 10 and Windows Server 2019](https://learn.microsoft.com/en-us/powershell/windows/get-started?view=windowsserver2019-ps)
* [Windows 11 and Windows Server 2022](https://learn.microsoft.com/en-us/powershell/windows/get-started?view=windowsserver2022-ps)

用以描述通过 PowerShell 命令行来对 Windows 进行功能设置和管理

The Windows PowerShell modules in the list support automating the features of those versions of the Windows operating system and provide links to the cmdlet references for each module. These modules enable you to use Windows PowerShell to administer, maintain, configure, and develop new features for Windows Server 2022 and Windows 11.


同 linux 中的命令命名逻辑不太一样, 同一个系列的不同命令 (增删改查) 的对应区别是由前缀决定的, 因此可以在一定程度上从后半部分的名称来对命令集进行分类

# NetSecurity


Network Security cmdlets

网络安全



## NetFirewallRule

命令包括:
* Copy-NetFirewallRule          : Copies an entire firewall rule, and associated filters, to the same or to a different policy store.
* Disable-NetFirewallRule       : Disables a firewall rule.
* Enable-NetFirewallRule        : Enables a previously disabled firewall rule.
* Get-NetFirewallRule 	        : Retrieves firewall rules from the target computer.
* New-NetFirewallRule           : Creates a new inbound or outbound firewall rule and adds the rule to the target computer.
* Remove-NetFirewallRule        : Deletes one or more firewall rules that match the specified criteria.
* Rename-NetFirewallRule        : Renames a single IPsec rule.
* Set-NetFirewallRule           : Modifies existing firewall rules.
* Show-NetFirewallRule          : Displays all of the existing firewall rules and associated objects in a fully expanded view.


防火墙规则的创建 - Creates a new inbound or outbound firewall rule and adds the rule to the target computer.



