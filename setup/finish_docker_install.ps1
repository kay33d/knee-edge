# Run in a normal PowerShell after restarting Windows.
# Puts Ubuntu, and therefore all Docker images and containers, on D:\wsl\ubuntu.
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = 1
$distro = 'Ubuntu-24.04'

New-Item -ItemType Directory -Force 'D:\wsl' | Out-Null
$installed = (wsl.exe -l -q) -join ' '
if ($installed -notmatch [regex]::Escape($distro)) {
    wsl.exe --install $distro --location 'D:\wsl\ubuntu' --no-launch
    if ($LASTEXITCODE -ne 0) { throw "Ubuntu install failed with exit code $LASTEXITCODE" }
    # wsl --install can exit 0 after only enabling a Windows component, so confirm the distro exists.
    if (((wsl.exe -l -q) -join ' ') -notmatch [regex]::Escape($distro)) {
        throw "Ubuntu was not installed. Windows probably needs a full restart: run 'shutdown /r /t 0', then rerun this script."
    }
}

$script = '/mnt/d/claude files/knee-edge/setup/install_docker_in_wsl.sh'
wsl.exe -d $distro -u root -- bash $script
if ($LASTEXITCODE -ne 0) { throw "Docker install failed with exit code $LASTEXITCODE" }

wsl.exe --shutdown
wsl.exe -d $distro -u root -- bash -c 'sleep 5; docker run --rm hello-world && docker info --format "root={{.DockerRootDir}} cpus={{.NCPU}} mem={{.MemTotal}}"; df -h /'
if ($LASTEXITCODE -ne 0) { throw "Docker test failed with exit code $LASTEXITCODE" }
Write-Output 'Docker is ready.'
Get-ChildItem 'D:\wsl\ubuntu' | Select-Object Name, @{n = 'GB'; e = { [math]::Round($_.Length / 1GB, 2) } }
