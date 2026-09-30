# 错误码速查

## ERR_1042

ERR_1042 表示数据库连接被拒绝。通常原因是数据库服务未启动，或
`config/db.yaml` 里的 `host` 与 `port` 写错。排查步骤：

1. `systemctl status mysql` 确认服务运行；
2. 用 `mysql -h <host> -P <port> -u root -p` 尝试手动连接；
3. 检查防火墙是否放行端口。

## ERR_2077

ERR_2077 表示上游服务超时。默认超时 3000ms，可在配置中调整 `upstream_timeout_ms`。
若持续出现，请检查上游健康状态。

## ERR_3310

ERR_3310 是权限不足。请确认访问令牌 scope 包含 `read:data`。
