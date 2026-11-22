# 用法：先在本会话外创建好仓库并准备 PAT（勾选 repo 权限），然后
#   $env:GITHUB_TOKEN = "<你的 PAT>"
#   powershell -ExecutionPolicy Bypass -File .\push.ps1
# 脚本不清除、不保存 token；只在本进程命令行内使用一次，不写入 .git/config。
param([string]$Repo = "https://github.com/echo060702/cv-project.git", [string]$Branch = "master")

$git = "C:\Program Files\Git\cmd\git.exe"
$token = $env:GITHUB_TOKEN
if (-not $token) { Write-Error "请先设置环境变量 GITHUB_TOKEN"; exit 1 }

# 本机 schannel 会报 SEC_E_NO_CREDENTIALS，用 openssl 后端绕过
$url = $Repo -replace '^https://', "https://x-access-token:$token@"
& $git -c http.sslBackend=openssl -c http.schannelCheckRevoke=false push -u $url $Branch
if ($LASTEXITCODE -ne 0) { Write-Error "推送失败：仓库不存在、无权限或 token 无效"; exit $LASTEXITCODE }
Write-Host "推送完成。请检查远端仓库的提交与文件列表。" -ForegroundColor Green
