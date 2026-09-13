param([ValidateRange(1,65535)][int]$Port = 8080)
$ErrorActionPreference = 'Stop'
$listener = $null
$lastError = $null
for ($candidate = $Port; $candidate -le [Math]::Min(65535, $Port + 19); $candidate++) {
  # Start() failure can dispose a listener. Never reuse that instance.
  $trial = New-Object System.Net.HttpListener
  try {
    $trial.Prefixes.Add(('http://localhost:' + $candidate + '/'))
    $trial.Start()
    $listener = $trial
    $Port = $candidate
    break
  } catch {
    $lastError = $_.Exception.Message
    try { $trial.Close() } catch {}
  }
}
if ($null -eq $listener) {
  Write-Error ('HTTP service could not start: ' + $lastError)
  exit 1
}
Write-Host ('Snake DQN UI: http://localhost:' + $Port + '/')
try {
  while ($listener.IsListening) {
    $ctx = $listener.GetContext()
    try {
      $path = $ctx.Request.Url.AbsolutePath
      $name = switch ($path) {
        '/' { 'index.html' }
        '/index.html' { 'index.html' }
        '/dqn.js' { 'dqn.js' }
        '/app.js' { 'app.js' }
        default { $null }
      }
      if ($null -eq $name) { $ctx.Response.StatusCode = 404 }
      else {
        $bytes = [IO.File]::ReadAllBytes((Join-Path $PSScriptRoot $name))
        if ($name.EndsWith('.js')) { $ctx.Response.ContentType = 'text/javascript; charset=utf-8' }
        else { $ctx.Response.ContentType = 'text/html; charset=utf-8' }
        $ctx.Response.Headers['Cache-Control'] = 'no-store'
        $ctx.Response.ContentLength64 = $bytes.Length
        $ctx.Response.OutputStream.Write($bytes, 0, $bytes.Length)
      }
    } finally { $ctx.Response.Close() }
  }
} finally {
  if ($null -ne $listener) { $listener.Close() }
}
